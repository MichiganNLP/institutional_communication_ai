#!/usr/bin/env python3
"""Audit whether requester-side AI assistance behaves differently by ACS context.

The audit joins completed Research Lines 6 and 8 outputs to the source sample.
It does not call an LLM and does not require a GPU.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


CONTEXTS = {
    "median_household_income_q": ("Income", 4),
    "poverty_rate_q": ("Poverty", 1),
    "unemployment_rate_q": ("Unemployment", 1),
    "bachelors_plus_share_q": ("Bachelor's degree", 4),
    "limited_english_household_share_q": ("Limited-English households", 1),
    "broadband_share_q": ("Broadband", 4),
    "homeownership_share_q": ("Homeownership", 4),
    "renter_share_q": ("Renter share", 1),
}

LINE8_METRICS = {
    "valid_json": "higher_is_better",
    "question_count": "descriptive_burden",
    "semantic_similarity": "higher_is_better",
    "lexical_jaccard": "descriptive_preservation",
    "unsupported_number_flag": "higher_is_riskier",
    "diagnostic_information_change": "higher_is_more_information",
    "surface_change": "higher_is_more_polish",
    "first_person_retention": "higher_is_more_voice_retention",
    "affect_retention": "higher_is_more_voice_retention",
}

LINE6_METRICS = {
    "semantic_similarity": "higher_is_better",
    "lexical_jaccard": "descriptive_preservation",
    "unsupported_number_flag": "higher_is_riskier",
    "surface_change": "higher_is_more_polish",
    "predicted_response_change": "higher_is_larger_predicted_return",
}

LINE8_CONTRASTS = {
    "targeted_minus_polish": ("targeted_elicitation", "polish_only"),
    "full_support_minus_polish": ("full_constrained_support", "polish_only"),
    "requirements_minus_polish": ("requirements_only", "polish_only"),
}

LINE6_CONTRASTS = {
    "professional_minus_voice_preserving": ("professional_register", "voice_preserving"),
    "plain_language_minus_professional": ("plain_language", "professional_register"),
}


def canonical_id(value: pd.Series) -> pd.Series:
    """Normalize IDs read once as integers and elsewhere as floating point."""
    return value.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def quartile_number(value: pd.Series) -> pd.Series:
    return pd.to_numeric(value.astype(str).str.extract(r"([1-4])", expand=False), errors="coerce")


def bool_number(value: pd.Series) -> pd.Series:
    text = value.astype(str).str.strip().str.lower()
    return text.map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})


def bootstrap_gap(low: np.ndarray, high: np.ndarray, draws: int, rng: np.random.Generator):
    low = low[np.isfinite(low)]
    high = high[np.isfinite(high)]
    if len(low) < 2 or len(high) < 2:
        return np.nan, np.nan, np.nan
    estimates = np.empty(draws)
    for index in range(draws):
        estimates[index] = (
            rng.choice(high, size=len(high), replace=True).mean()
            - rng.choice(low, size=len(low), replace=True).mean()
        )
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    p_value = min(1.0, 2 * min(np.mean(estimates <= 0), np.mean(estimates >= 0)))
    return float(lower), float(upper), float(p_value)


def add_context(records: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    records = records.copy()
    sample = sample.copy()
    records["record_id"] = canonical_id(records["record_id"])
    sample["record_id"] = canonical_id(sample["record_id"])
    available = [
        "record_id", "actionability", "log_words", "post_chatgpt", "category",
        "institution", "month", "month_of_year", *CONTEXTS,
    ]
    metadata = sample[[column for column in available if column in sample]].drop_duplicates("record_id")
    result = records.merge(metadata, on="record_id", how="left", validate="many_to_one", suffixes=("", "_sample"))
    for column in CONTEXTS:
        if column in result:
            result[column] = quartile_number(result[column])
    return result


def summarize_cells(records, group_column, metrics, contexts):
    rows = []
    for context in contexts:
        if context not in records:
            continue
        label, advantaged_q = CONTEXTS[context]
        for (group, quartile), part in records.groupby([group_column, context], dropna=False):
            if not np.isfinite(quartile):
                continue
            for metric, direction in metrics.items():
                values = pd.to_numeric(part.get(metric), errors="coerce").dropna()
                rows.append({
                    "context_variable": context,
                    "context_label": label,
                    "advantaged_quartile": advantaged_q,
                    group_column: group,
                    "quartile": int(quartile),
                    "metric": metric,
                    "metric_direction": direction,
                    "n": len(values),
                    "mean": values.mean() if len(values) else np.nan,
                    "sd": values.std(ddof=1) if len(values) > 1 else np.nan,
                })
    return pd.DataFrame(rows)


def summarize_extreme_gaps(records, group_column, metrics, contexts, draws, seed):
    rows = []
    rng = np.random.default_rng(seed)
    for context in contexts:
        if context not in records:
            continue
        label, advantaged_q = CONTEXTS[context]
        for group, part in records.groupby(group_column):
            for metric, direction in metrics.items():
                values = pd.to_numeric(part.get(metric), errors="coerce")
                low = values[part[context].eq(1)].dropna().to_numpy(float)
                high = values[part[context].eq(4)].dropna().to_numpy(float)
                ci_low, ci_high, p_value = bootstrap_gap(low, high, draws, rng)
                high_minus_low = high.mean() - low.mean() if len(low) and len(high) else np.nan
                orientation = 1 if advantaged_q == 4 else -1
                rows.append({
                    "context_variable": context,
                    "context_label": label,
                    "advantaged_quartile": advantaged_q,
                    group_column: group,
                    "metric": metric,
                    "metric_direction": direction,
                    "n_low_q1": len(low),
                    "n_high_q4": len(high),
                    "mean_low_q1": low.mean() if len(low) else np.nan,
                    "mean_high_q4": high.mean() if len(high) else np.nan,
                    "high_minus_low": high_minus_low,
                    "high_minus_low_ci_low": ci_low,
                    "high_minus_low_ci_high": ci_high,
                    "high_minus_low_p": p_value,
                    "advantaged_minus_disadvantaged": orientation * high_minus_low,
                    "advantaged_minus_disadvantaged_ci_low": orientation * (ci_low if orientation == 1 else ci_high),
                    "advantaged_minus_disadvantaged_ci_high": orientation * (ci_high if orientation == 1 else ci_low),
                })
    return pd.DataFrame(rows)


def paired_contrast_records(records, group_column, contrasts, metrics, valid_only=False):
    rows = []
    base_columns = ["record_id", group_column]
    for metric in metrics:
        work = records.copy()
        if valid_only and metric != "valid_json":
            validity_column = "usable_revision" if "usable_revision" in work else "valid_json"
            work.loc[~work[validity_column].eq(1), metric] = np.nan
        wide = work.pivot_table(index="record_id", columns=group_column, values=metric, aggfunc="first")
        for contrast, (left, right) in contrasts.items():
            if left not in wide or right not in wide:
                continue
            values = (wide[left] - wide[right]).rename("contrast_value").reset_index()
            values["contrast"] = contrast
            values["metric"] = metric
            rows.append(values)
    if not rows:
        return pd.DataFrame(columns=["record_id", "contrast", "metric", "contrast_value"])
    output = pd.concat(rows, ignore_index=True)
    metadata_columns = [
        "record_id", "actionability", "log_words", "post_chatgpt", "category",
        "institution", "month", "month_of_year", *CONTEXTS,
    ]
    metadata = records[[column for column in metadata_columns if column in records]].drop_duplicates("record_id")
    return output.merge(metadata, on="record_id", how="left", validate="many_to_one")


def summarize_contrasts(records, metrics, contexts, draws, seed):
    renamed = records.rename(columns={"contrast": "policy_contrast"})
    directions = {metric: direction for metric, direction in metrics.items()}
    rows = []
    rng = np.random.default_rng(seed)
    for context in contexts:
        if context not in renamed:
            continue
        label, advantaged_q = CONTEXTS[context]
        orientation = 1 if advantaged_q == 4 else -1
        for (contrast, metric), part in renamed.groupby(["policy_contrast", "metric"]):
            values = pd.to_numeric(part["contrast_value"], errors="coerce")
            low = values[part[context].eq(1)].dropna().to_numpy(float)
            high = values[part[context].eq(4)].dropna().to_numpy(float)
            ci_low, ci_high, p_value = bootstrap_gap(low, high, draws, rng)
            raw_gap = high.mean() - low.mean() if len(low) and len(high) else np.nan
            rows.append({
                "context_variable": context,
                "context_label": label,
                "advantaged_quartile": advantaged_q,
                "policy_contrast": contrast,
                "metric": metric,
                "metric_direction": directions.get(metric, "descriptive"),
                "n_low_q1": len(low),
                "n_high_q4": len(high),
                "mean_contrast_low_q1": low.mean() if len(low) else np.nan,
                "mean_contrast_high_q4": high.mean() if len(high) else np.nan,
                "difference_in_contrasts_high_minus_low": raw_gap,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bootstrap_p": p_value,
                "difference_in_contrasts_advantaged_minus_disadvantaged": orientation * raw_gap,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cfpb", "sandiego"], required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--line6", required=True)
    parser.add_argument("--line8", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--contexts", nargs="+", default=list(CONTEXTS), choices=list(CONTEXTS),
        help="ACS quartile columns to audit (default: all available columns).",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(args.sample, low_memory=False)

    line8 = add_context(pd.read_csv(args.line8, low_memory=False), sample)
    line8["valid_json"] = bool_number(line8["valid_json"])
    if "usable_revision" in line8:
        line8["usable_revision"] = bool_number(line8["usable_revision"])
    else:
        line8["usable_revision"] = line8["valid_json"]
    line8["unsupported_number_flag"] = pd.to_numeric(
        line8["unsupported_number_count"], errors="coerce"
    ).gt(0).astype(float)
    valid = line8["usable_revision"].eq(1)
    for metric in LINE8_METRICS:
        if metric != "valid_json":
            line8.loc[~valid, metric] = np.nan

    line8_cells = summarize_cells(line8, "policy", LINE8_METRICS, args.contexts)
    line8_gaps = summarize_extreme_gaps(
        line8, "policy", LINE8_METRICS, args.contexts, args.bootstrap, args.seed
    )
    line8_contrast_records = paired_contrast_records(
        line8, "policy", LINE8_CONTRASTS, LINE8_METRICS, valid_only=True
    )
    line8_contrasts = summarize_contrasts(
        line8_contrast_records, LINE8_METRICS, args.contexts, args.bootstrap, args.seed + 1
    )

    line6 = add_context(pd.read_csv(args.line6, low_memory=False), sample)
    line6["unsupported_number_flag"] = pd.to_numeric(
        line6["unsupported_number_count"], errors="coerce"
    ).gt(0).astype(float)
    line6_cells = summarize_cells(line6, "condition", LINE6_METRICS, args.contexts)
    line6_gaps = summarize_extreme_gaps(
        line6, "condition", LINE6_METRICS, args.contexts, args.bootstrap, args.seed + 2
    )
    line6_contrast_records = paired_contrast_records(
        line6, "condition", LINE6_CONTRASTS, LINE6_METRICS
    )
    line6_contrasts = summarize_contrasts(
        line6_contrast_records, LINE6_METRICS, args.contexts, args.bootstrap, args.seed + 3
    )

    outputs = {
        "line8_policy_quartile_cells.csv": line8_cells,
        "line8_policy_extreme_gaps.csv": line8_gaps,
        "line8_paired_policy_contrast_records.csv.gz": line8_contrast_records,
        "line8_paired_policy_contrasts.csv": line8_contrasts,
        "line6_condition_quartile_cells.csv": line6_cells,
        "line6_condition_extreme_gaps.csv": line6_gaps,
        "line6_paired_condition_contrast_records.csv.gz": line6_contrast_records,
        "line6_paired_condition_contrasts.csv": line6_contrasts,
    }
    for filename, frame in outputs.items():
        frame.insert(0, "dataset", args.dataset)
        frame.to_csv(outdir / filename, index=False, compression="gzip" if filename.endswith(".gz") else None)

    manifest = {
        "dataset": args.dataset,
        "sample_rows": int(len(sample)),
        "line6_rows": int(len(line6)),
        "line8_rows": int(len(line8)),
        "line8_valid_json_rate": float(line8["valid_json"].mean()),
        "line8_usable_revision_rate": float(line8["usable_revision"].mean()),
        "contexts": args.contexts,
        "bootstrap_draws": args.bootstrap,
        "seed": args.seed,
        "estimand": "Q4 minus Q1; oriented columns reverse signs when Q1 is the advantaged reference",
        "validity_note": "Rewrite metrics use nonempty usable revisions; valid_json is reported separately. Plain-text recovery, when enabled in Line 8 analysis, is not JSON compliance.",
    }
    (outdir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote neighborhood assistance audit for {args.dataset} to {outdir}", flush=True)


if __name__ == "__main__":
    main()
