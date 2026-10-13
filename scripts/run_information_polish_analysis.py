#!/usr/bin/env python3
"""Separate diagnostic information from surface presentation in institutional requests.

The pipeline has two subcommands:

1. ``build`` streams the original dataset, applies the corrected fixed pre-period
   information score, learns a fixed pre-period surface-presentation scale, and
   writes compact record shards without raw text.
2. ``model`` estimates production changes and independent response returns for
   diagnostic information and surface presentation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from absorbed_linear import fit_absorbed_ols
from corrected_actionability_common import (
    COMPONENTS,
    DATASETS,
    apply_scores,
    ensure_directory,
    extract_base,
    fit_reference_parameters,
    resolve_columns,
    text_features,
)


OUTCOMES = ["response_favorable", "response_timely", "response_complete"]
SURFACE_COMPONENTS = [
    "sentence_organization",
    "paragraph_structure",
    "casing_control",
    "punctuation_control",
    "formality_control",
    "lexical_complexity",
]
PRIMARY_OUTCOME = {
    "cfpb": "response_favorable",
    "sandiego": "response_timely",
    "foia": "response_favorable",
}
NORMALIZED_ALIASES = {
    "cfpb": {
        "id": ["complaint_id"],
        "date": ["date_received"],
        "text": ["complaint_what_happened"],
        "category": ["product"],
        "subcategory": ["issue", "sub_issue"],
        "institution": ["company"],
        "context": ["submitted_via"],
        "state": ["state"],
        "zip": ["zip_code"],
        "status": ["company_response"],
        "timely": ["timely"],
    }
}

WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")
SENTENCE_END_RE = re.compile(r"[.!?]+(?:\s|$)")
PARAGRAPH_BREAK_RE = re.compile(r"(?:\r?\n\s*){2,}")
UPPER_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")
LONG_WORD_RE = re.compile(r"\b[A-Za-z]{7,}\b")
REPEATED_PUNCT_RE = re.compile(r"[!?.,]{2,}")
INFORMAL_RE = re.compile(
    r"\b(?:ain't|can't|couldn't|didn't|doesn't|don't|gonna|gotta|hadn't|hasn't|"
    r"haven't|idk|isn't|kinda|lol|shouldn't|wasn't|weren't|won't|wanna|wouldn't)\b",
    re.I,
)


def read_chunks(path, usecols, chunksize):
    return pd.read_csv(path, dtype=str, usecols=usecols, chunksize=chunksize, low_memory=False)


def surface_features(text: pd.Series, word_count: pd.Series | None = None) -> pd.DataFrame:
    """Return transparent, bounded document-level surface-presentation features."""
    normalized = text.fillna("").astype(str).str.strip()
    words = word_count if word_count is not None else normalized.str.count(WORD_RE).astype(float)
    words = pd.to_numeric(words, errors="coerce").fillna(0.0).astype(float)
    denominator = words.clip(lower=20.0)

    sentence_count = normalized.str.count(SENTENCE_END_RE).clip(lower=1).astype(float)
    mean_sentence_words = words.clip(lower=1.0) / sentence_count
    # Highest at roughly 18 words per sentence; symmetric on a ratio scale.
    sentence_organization = np.exp(-np.abs(np.log(mean_sentence_words.clip(lower=1.0) / 18.0)))

    paragraph_count = normalized.str.count(PARAGRAPH_BREAK_RE).astype(float) + 1.0
    expected_paragraphs = (words / 100.0).clip(lower=1.0)
    paragraph_structure = np.minimum(paragraph_count / expected_paragraphs, 1.0)

    uppercase_share = normalized.str.count(UPPER_WORD_RE).astype(float) / denominator
    casing_control = np.exp(-10.0 * uppercase_share)

    repeated_punctuation = normalized.str.count(REPEATED_PUNCT_RE).astype(float)
    punctuation_control = np.exp(-5.0 * repeated_punctuation / sentence_count)

    informal_share = normalized.str.count(INFORMAL_RE).astype(float) / denominator
    formality_control = np.exp(-10.0 * informal_share)

    lexical_complexity = normalized.str.count(LONG_WORD_RE).astype(float) / denominator

    return pd.DataFrame(
        {
            "sentence_organization": sentence_organization,
            "paragraph_structure": paragraph_structure,
            "casing_control": casing_control,
            "punctuation_control": punctuation_control,
            "formality_control": formality_control,
            "lexical_complexity": lexical_complexity,
        },
        index=text.index,
    )


def eligible(base, features, start, end, min_words):
    return (
        base["opened"].ge(start)
        & base["opened"].lt(end)
        & base["text"].ne("")
        & features["word_count"].ge(min_words)
    )


def apply_fixed_z(features, parameters, prefix):
    result = pd.DataFrame(index=features.index)
    for name in parameters:
        parameter = parameters[name]
        values = pd.to_numeric(features[name], errors="coerce").clip(upper=parameter["cap"])
        result[f"{prefix}_{name}_z"] = (values - parameter["mean"]) / parameter["sd"]
    return result


def collapse_levels(series, retained):
    if not retained:
        return series.fillna("Missing").astype(str)
    normalized = series.fillna("Missing").astype(str)
    return normalized.where(normalized.isin(set(retained)), "Other")


def build(args):
    dataset = args.dataset
    input_path = Path(args.input)
    corrected_paneldir = Path(args.corrected_paneldir)
    outdir = Path(args.outdir)
    shard_dir = outdir / "analytic_records"
    ensure_directory(outdir)
    ensure_directory(shard_dir)

    corrected_metadata_path = corrected_paneldir / f"{dataset}_actionability_parameters.json"
    if not corrected_metadata_path.exists():
        raise FileNotFoundError(f"Missing corrected score metadata: {corrected_metadata_path}")
    corrected_metadata = json.loads(corrected_metadata_path.read_text())
    information_parameters = corrected_metadata["feature_parameters"]
    retained_levels = corrected_metadata.get("kept_levels", {})

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    cutoff = pd.Timestamp(args.cutoff)
    header = list(pd.read_csv(input_path, nrows=0).columns)
    aliases = NORMALIZED_ALIASES.get(dataset, {})
    dataset_spec = {
        key: [*values, *aliases.get(key, [])]
        for key, values in DATASETS[dataset].items()
    }
    mapping = resolve_columns(header, dataset_spec)
    if not mapping["date"] or not mapping["text"]:
        raise KeyError(f"Missing date or text columns. Resolved mapping: {mapping}")
    usecols = sorted({column for columns in mapping.values() for column in columns})

    surface_arrays = {name: [] for name in SURFACE_COMPONENTS}
    print(f"{dataset} surface pass 1: learn pre-period reference scale", flush=True)
    for index, chunk in enumerate(read_chunks(input_path, usecols, args.chunksize), start=1):
        base = extract_base(chunk, mapping, dataset, end)
        information = text_features(base["text"])
        keep = eligible(base, information, start, end, args.min_words)
        base = base.loc[keep]
        information = information.loc[keep]
        surface = surface_features(base["text"], information["word_count"])
        pre = base["opened"].lt(cutoff)
        for name in SURFACE_COMPONENTS:
            surface_arrays[name].append(surface.loc[pre, name].to_numpy(dtype=np.float32))
        print(f"surface pass 1 chunk {index}: {len(base):,} eligible; {int(pre.sum()):,} pre", flush=True)

    surface_parameters = fit_reference_parameters(surface_arrays)
    del surface_arrays
    analysis_metadata = {
        "dataset": dataset,
        "input": str(input_path),
        "corrected_paneldir": str(corrected_paneldir),
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "cutoff": str(cutoff.date()),
        "min_words": args.min_words,
        "diagnostic_information_components": COMPONENTS,
        "surface_components": SURFACE_COMPONENTS,
        "surface_reference_population": "eligible pre-cutoff requester-authored narratives",
        "surface_feature_parameters": surface_parameters,
        "surface_polish_primary": "mean pre-period z-score of all surface components",
        "surface_polish_robustness": "mean excluding lexical_complexity",
        "raw_text_retained": False,
    }
    (outdir / f"{dataset}_information_polish_parameters.json").write_text(
        json.dumps(analysis_metadata, indent=2) + "\n"
    )

    manifest_paths = []
    summary_parts = []
    print(f"{dataset} surface pass 2: score and write compact shards", flush=True)
    for index, chunk in enumerate(read_chunks(input_path, usecols, args.chunksize), start=1):
        shard_path = shard_dir / f"records_{index:05d}.csv.gz"
        manifest_paths.append(shard_path.resolve())
        if shard_path.exists() and not args.force:
            print(f"surface pass 2 chunk {index}: shard exists", flush=True)
            continue

        base = extract_base(chunk, mapping, dataset, end)
        information_raw = text_features(base["text"])
        keep = eligible(base, information_raw, start, end, args.min_words)
        base = base.loc[keep].copy()
        information_raw = information_raw.loc[keep]
        information_scored = apply_scores(information_raw, information_parameters)
        component_parameters = {name: information_parameters[name] for name in COMPONENTS}
        information_z = apply_fixed_z(information_raw[COMPONENTS], component_parameters, "information")
        surface_raw = surface_features(base["text"], information_raw["word_count"])
        surface_z = apply_fixed_z(surface_raw, surface_parameters, "surface")

        frame = base.drop(columns=["text", "opened"]).copy()
        frame["information_score"] = information_scored["actionability"]
        frame["log_words"] = information_raw["log_words"]
        frame["word_count"] = information_raw["word_count"]
        frame = pd.concat([frame, information_z, surface_raw, surface_z], axis=1)
        surface_z_columns = [f"surface_{name}_z" for name in SURFACE_COMPONENTS]
        no_lexical_columns = [name for name in surface_z_columns if name != "surface_lexical_complexity_z"]
        frame["surface_polish"] = frame[surface_z_columns].mean(axis=1)
        frame["surface_polish_no_lexical"] = frame[no_lexical_columns].mean(axis=1)

        opened = base["opened"]
        frame["post_chatgpt"] = opened.ge(cutoff).astype(np.int8)
        frame["month"] = opened.dt.to_period("M").astype(str)
        frame["month_of_year"] = opened.dt.month.astype(np.int8)
        frame["time_index"] = ((opened.dt.year - start.year) * 12 + opened.dt.month - start.month).astype(np.int16)
        for name in ["category", "subcategory", "institution", "context"]:
            frame[name] = collapse_levels(frame[name], retained_levels.get(name, []))

        output_columns = [
            "record_id",
            "post_chatgpt",
            "month",
            "month_of_year",
            "time_index",
            "category",
            "subcategory",
            "institution",
            "context",
            "state",
            "zip",
            "information_score",
            "surface_polish",
            "surface_polish_no_lexical",
            "log_words",
            "word_count",
            *[f"information_{name}_z" for name in COMPONENTS],
            *SURFACE_COMPONENTS,
            *surface_z_columns,
            *OUTCOMES,
        ]
        frame[output_columns].to_csv(shard_path, index=False, compression="gzip")

        summary = frame.groupby("post_chatgpt", dropna=False).agg(
            n=("record_id", "size"),
            information_score_sum=("information_score", "sum"),
            surface_polish_sum=("surface_polish", "sum"),
            surface_polish_no_lexical_sum=("surface_polish_no_lexical", "sum"),
        ).reset_index()
        summary_parts.append(summary)
        print(f"surface pass 2 chunk {index}: wrote {len(frame):,} records", flush=True)

    manifest = outdir / f"{dataset}_information_polish_manifest.txt"
    manifest.write_text("\n".join(str(path) for path in manifest_paths) + "\n")
    if summary_parts:
        summary = pd.concat(summary_parts, ignore_index=True).groupby("post_chatgpt", as_index=False).sum()
        for score in ["information_score", "surface_polish", "surface_polish_no_lexical"]:
            summary[score] = summary[f"{score}_sum"] / summary["n"]
        summary.to_csv(outdir / f"{dataset}_information_polish_build_summary.csv", index=False)
    print(f"Wrote manifest: {manifest}", flush=True)


def load_records(paneldir, dataset, max_records=None):
    manifest = paneldir / f"{dataset}_information_polish_manifest.txt"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing build manifest: {manifest}")
    paths = [Path(line) for line in manifest.read_text().splitlines() if line.strip()]
    parts = []
    remaining = max_records
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Manifest shard does not exist: {path}")
        part = pd.read_csv(path, low_memory=False)
        if remaining is not None:
            part = part.head(remaining)
            remaining -= len(part)
        if len(part):
            parts.append(part)
        if remaining is not None and remaining <= 0:
            break
    if not parts:
        raise ValueError("No analytic records found")
    records = pd.concat(parts, ignore_index=True)
    for column in ["category", "subcategory", "institution", "context", "state", "month"]:
        if column in records:
            records[column] = records[column].fillna("Missing").astype("category")
    return records


def write_result(result, path, metadata):
    table = result.table()
    for key, value in metadata.items():
        table[key] = value
    table["n"] = result.n
    table["r2_within"] = result.r2_within
    table["absorption_iterations"] = result.iterations
    table["cluster_count"] = result.cluster_count
    table.to_csv(path, index=False)
    return table


def available_fixed_effects(frame, candidates):
    return [name for name in candidates if name in frame and frame[name].nunique(dropna=False) > 1]


def add_post_interactions(frame, variables):
    result = frame.copy()
    for variable in variables:
        result[f"{variable}_post"] = pd.to_numeric(result[variable], errors="coerce") * result["post_chatgpt"]
    return result


def run_production_models(records, dataset, outdir):
    data = records.copy()
    first_post = int(data.loc[data["post_chatgpt"].eq(1), "time_index"].min())
    data["time_after"] = np.maximum(data["time_index"] - first_post + 1, 0.0)
    fixed_effects = available_fixed_effects(data, ["category", "institution", "month_of_year"])
    outputs = []
    for score in ["information_score", "surface_polish", "surface_polish_no_lexical"]:
        result = fit_absorbed_ols(
            data,
            score,
            ["time_index", "post_chatgpt", "time_after", "log_words"],
            fixed_effects,
            cluster="month",
        )
        outputs.append(
            write_result(
                result,
                outdir / f"production_{score}.csv",
                {
                    "dataset": dataset,
                    "analysis": "information_polish_production_its",
                    "score": score,
                    "fixed_effects": "+".join(fixed_effects),
                },
            )
        )
    pd.concat(outputs, ignore_index=True).to_csv(outdir / "production_model_index.csv", index=False)


def model_specifications():
    return {
        "information_only": ["information_score"],
        "surface_only": ["surface_polish"],
        "joint": ["information_score", "surface_polish"],
        "joint_surface_no_lexical": ["information_score", "surface_polish_no_lexical"],
    }


def run_response_models(records, dataset, outcomes, outdir, skip_components=False):
    fixed_effects = available_fixed_effects(
        records, ["category", "subcategory", "month", "institution", "context", "state"]
    )
    all_tables = []
    marginal_rows = []
    comparison_rows = []
    for outcome in outcomes:
        if (
            outcome not in records
            or records[outcome].notna().sum() < 100
            or records[outcome].dropna().nunique() < 2
        ):
            continue
        for model_name, focal_variables in model_specifications().items():
            variables = [*focal_variables, "log_words"]
            data = add_post_interactions(records, variables)
            x = [item for variable in variables for item in [variable, f"{variable}_post"]]
            result = fit_absorbed_ols(data, outcome, x, fixed_effects, cluster="institution")
            metadata = {
                "dataset": dataset,
                "analysis": "information_vs_surface_response",
                "outcome": outcome,
                "model": model_name,
                "fixed_effects": "+".join(fixed_effects),
            }
            table = write_result(result, outdir / f"response_{outcome}_{model_name}.csv", metadata)
            all_tables.append(table)
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "outcome": outcome,
                    "model": model_name,
                    "n": result.n,
                    "r2_within": result.r2_within,
                    "fixed_effects": "+".join(fixed_effects),
                }
            )
            if model_name in {"joint", "joint_surface_no_lexical"}:
                for variable in focal_variables:
                    for post in [0, 1]:
                        estimate, se, pvalue = result.contrast(
                            {variable: 1.0, f"{variable}_post": float(post)}
                        )
                        marginal_rows.append(
                            {
                                "dataset": dataset,
                                "outcome": outcome,
                                "model": model_name,
                                "score": variable,
                                "post_chatgpt": post,
                                "marginal_return": estimate,
                                "se": se,
                                "pvalue": pvalue,
                                "n": result.n,
                            }
                        )

        if skip_components:
            continue
        component_variables = [
            *[f"information_{name}_z" for name in COMPONENTS],
            *[f"surface_{name}_z" for name in SURFACE_COMPONENTS],
        ]
        variables = [*component_variables, "log_words"]
        data = add_post_interactions(records, variables)
        x = [item for variable in variables for item in [variable, f"{variable}_post"]]
        result = fit_absorbed_ols(data, outcome, x, fixed_effects, cluster="institution")
        metadata = {
            "dataset": dataset,
            "analysis": "information_surface_component_response",
            "outcome": outcome,
            "model": "component_joint",
            "fixed_effects": "+".join(fixed_effects),
        }
        all_tables.append(
            write_result(result, outdir / f"response_{outcome}_component_joint.csv", metadata)
        )

    if all_tables:
        pd.concat(all_tables, ignore_index=True).to_csv(outdir / "response_model_index.csv", index=False)
    if marginal_rows:
        pd.DataFrame(marginal_rows).to_csv(outdir / "response_marginal_returns.csv", index=False)
    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(outdir / "response_model_comparison.csv", index=False)


def descriptive_summary(records, dataset, outdir):
    score_columns = [
        "information_score",
        "surface_polish",
        "surface_polish_no_lexical",
        *[f"information_{name}_z" for name in COMPONENTS],
        *[f"surface_{name}_z" for name in SURFACE_COMPONENTS],
        "log_words",
    ]
    rows = []
    for post, group in records.groupby("post_chatgpt"):
        for score in score_columns:
            values = pd.to_numeric(group[score], errors="coerce").dropna()
            rows.append(
                {
                    "dataset": dataset,
                    "post_chatgpt": int(post),
                    "score": score,
                    "n": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=0),
                    "p25": values.quantile(0.25),
                    "median": values.median(),
                    "p75": values.quantile(0.75),
                }
            )
    pd.DataFrame(rows).to_csv(outdir / "information_polish_descriptive_summary.csv", index=False)


def model(args):
    paneldir = Path(args.paneldir)
    outdir = Path(args.outdir)
    ensure_directory(outdir)
    records = load_records(paneldir, args.dataset, args.max_records)
    print(f"Loaded {len(records):,} {args.dataset} records", flush=True)
    descriptive_summary(records, args.dataset, outdir)
    run_production_models(records, args.dataset, outdir)
    requested = args.outcomes if args.outcomes else OUTCOMES
    run_response_models(records, args.dataset, requested, outdir, args.skip_components)
    print(f"Wrote information-versus-polish results to {outdir}", flush=True)


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    subparsers = main.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build compact information/polish shards")
    build_parser.add_argument("dataset", choices=sorted(DATASETS))
    build_parser.add_argument("--input", required=True)
    build_parser.add_argument("--corrected-paneldir", required=True)
    build_parser.add_argument("--outdir", required=True)
    build_parser.add_argument("--chunksize", type=int, default=75000)
    build_parser.add_argument("--start-date", default="2021-01-01")
    build_parser.add_argument("--end-date", default="2025-01-01")
    build_parser.add_argument("--cutoff", default="2022-11-30")
    build_parser.add_argument("--min-words", type=int, default=3)
    build_parser.add_argument("--force", action="store_true")
    build_parser.set_defaults(function=build)

    model_parser = subparsers.add_parser("model", help="Fit information-versus-polish models")
    model_parser.add_argument("dataset", choices=sorted(DATASETS))
    model_parser.add_argument("--paneldir", required=True)
    model_parser.add_argument("--outdir", required=True)
    model_parser.add_argument("--outcomes", nargs="*", choices=OUTCOMES)
    model_parser.add_argument("--max-records", type=int)
    model_parser.add_argument("--skip-components", action="store_true")
    model_parser.set_defaults(function=model)
    return main


def main():
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
