#!/usr/bin/env python3
"""Build globally scaled actionability scores and compact analytic cells.

The script uses three streaming passes:
1. learn winsorization/scaling parameters from pre-cutoff narratives;
2. learn the pre-period top-quartile threshold;
3. aggregate fixed-scale scores, components, outcomes, and ACS context.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from corrected_actionability_common import (
    ACS_VARS,
    CHATGPT_DATE,
    COMPONENTS,
    DATASETS,
    SCORE_COLUMNS,
    add_acs,
    apply_scores,
    coalesce,
    ensure_directory,
    extract_base,
    fit_reference_parameters,
    load_acs_quartiles,
    resolve_columns,
    text_features,
)


OUTCOMES = [
    "response_favorable",
    "response_timely",
    "response_complete",
    "response_hours",
    "response_event",
    "response_duration_hours",
]
FEATURE_OUTPUTS = [*COMPONENTS, "log_words", "word_count"]


def read_chunks(path, usecols, chunksize):
    return pd.read_csv(path, dtype=str, usecols=usecols, chunksize=chunksize, low_memory=False)


def keep_dates(base, start, end, min_words=None, features=None):
    keep = base["opened"].ge(start) & base["opened"].lt(end) & base["text"].ne("")
    if min_words is not None and features is not None:
        keep &= features["word_count"].ge(min_words)
    return keep


def top_values(counter, maximum):
    return {value for value, _ in counter.most_common(maximum)}


def collapse_levels(series, keep):
    return series.where(series.isin(keep), "Other")


def aggregate_chunk(frame, group_cols, numeric_cols):
    aggregations = {"n": ("record_id", "size")}
    for column in numeric_cols:
        aggregations[column + "_sum"] = (column, "sum")
        aggregations[column + "_n"] = (column, "count")
    return frame.groupby(group_cols, dropna=False).agg(**aggregations).reset_index()


def combine_checkpoints(paths, group_cols, numeric_cols):
    parts = [part for path in paths if len(part := pd.read_csv(path))]
    if not parts:
        raise ValueError("No eligible analytic cells were produced")
    raw = pd.concat(parts, ignore_index=True)
    sum_cols = [c + suffix for c in numeric_cols for suffix in ["_sum", "_n"]]
    agg = {"n": "sum", **{column: "sum" for column in sum_cols}}
    cells = raw.groupby(group_cols, dropna=False).agg(agg).reset_index()
    for column in numeric_cols:
        denominator = pd.to_numeric(cells[column + "_n"], errors="coerce").astype(float)
        cells[column] = cells[column + "_sum"] / denominator.mask(denominator.eq(0))
    return cells


def monthly_summary(cells, numeric_cols):
    rows = []
    for month, group in cells.groupby("month"):
        row = {"month": month, "n": int(group["n"].sum())}
        row["post_chatgpt"] = int(group["post_chatgpt"].iloc[0])
        row["time_index"] = int(group["time_index"].iloc[0])
        for column in numeric_cols:
            ncol = column + "_n"
            scol = column + "_sum"
            denominator = group[ncol].sum()
            row[column] = group[scol].sum() / denominator if denominator else np.nan
            row[ncol] = int(denominator)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--acs")
    parser.add_argument("--chunksize", type=int, default=100000)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2025-01-01")
    parser.add_argument("--cutoff", default="2022-11-30")
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-category-levels", type=int, default=60)
    parser.add_argument("--max-subcategory-levels", type=int, default=80)
    parser.add_argument("--max-institution-levels", type=int, default=100)
    parser.add_argument("--max-context-levels", type=int, default=60)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    checkpoint_dir = outdir / "_cell_checkpoints"
    record_dir = outdir / "analytic_records"
    ensure_directory(outdir)
    ensure_directory(checkpoint_dir)
    ensure_directory(record_dir)
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    cutoff = pd.Timestamp(args.cutoff)

    header = list(pd.read_csv(input_path, nrows=0).columns)
    mapping = resolve_columns(header, DATASETS[args.dataset])
    if not mapping["date"] or not mapping["text"]:
        raise KeyError(f"Missing date or text columns. Resolved mapping: {mapping}")
    usecols = sorted({column for columns in mapping.values() for column in columns})

    reference_arrays = {name: [] for name in [*COMPONENTS, "log_words"]}
    level_counts = {name: Counter() for name in ["category", "subcategory", "institution", "context"]}
    print(f"{args.dataset} pass 1: pre-period feature reference and level counts", flush=True)
    for index, chunk in enumerate(read_chunks(input_path, usecols, args.chunksize), start=1):
        base = extract_base(chunk, mapping, args.dataset, end)
        features = text_features(base["text"])
        keep = keep_dates(base, start, end, args.min_words, features)
        base = base.loc[keep]
        features = features.loc[keep]
        pre = base["opened"].lt(cutoff)
        for name in reference_arrays:
            reference_arrays[name].append(features.loc[pre, name].to_numpy(dtype=np.float32))
        for name in level_counts:
            level_counts[name].update(base[name])
        print(f"pass 1 chunk {index}: {len(base):,} eligible; {int(pre.sum()):,} pre", flush=True)

    params = fit_reference_parameters(reference_arrays)
    del reference_arrays
    keep_levels = {
        "category": top_values(level_counts["category"], args.max_category_levels),
        "subcategory": top_values(level_counts["subcategory"], args.max_subcategory_levels),
        "institution": top_values(level_counts["institution"], args.max_institution_levels),
        "context": top_values(level_counts["context"], args.max_context_levels),
    }

    print(f"{args.dataset} pass 2: fixed pre-period actionability threshold", flush=True)
    pre_scores = []
    for index, chunk in enumerate(read_chunks(input_path, usecols, args.chunksize), start=1):
        base = extract_base(chunk, mapping, args.dataset, end)
        features = text_features(base["text"])
        keep = keep_dates(base, start, end, args.min_words, features)
        base = base.loc[keep]
        features = features.loc[keep]
        scored = apply_scores(features, params)
        pre_scores.append(scored.loc[base["opened"].lt(cutoff), "actionability"].to_numpy(dtype=np.float32))
        print(f"pass 2 chunk {index}: {len(scored):,} eligible", flush=True)
    high_threshold = float(np.quantile(np.concatenate(pre_scores), 0.75))
    del pre_scores

    metadata = {
        "dataset": args.dataset,
        "input": str(input_path),
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "cutoff": str(cutoff.date()),
        "min_words": args.min_words,
        "mapping": mapping,
        "components": COMPONENTS,
        "primary_score": "actionability",
        "primary_score_includes_length": False,
        "reference_population": "eligible pre-cutoff narratives",
        "feature_parameters": params,
        "preperiod_top_quartile_threshold": high_threshold,
        "kept_levels": {key: sorted(values) for key, values in keep_levels.items()},
    }
    (outdir / f"{args.dataset}_actionability_parameters.json").write_text(json.dumps(metadata, indent=2))

    acs = load_acs_quartiles(args.acs)
    qcols = [variable + "_q" for variable in ACS_VARS if acs is not None and variable + "_q" in acs.columns]
    group_cols = [
        "post_chatgpt",
        "month",
        "month_of_year",
        "time_index",
        "category",
        "subcategory",
        "institution",
        "context",
        "state",
        *qcols,
    ]
    numeric_cols = [*SCORE_COLUMNS, "top_quartile_actionability", *FEATURE_OUTPUTS, *OUTCOMES]

    print(f"{args.dataset} pass 3: analytic-cell aggregation", flush=True)
    checkpoint_paths = []
    record_paths = []
    for index, chunk in enumerate(read_chunks(input_path, usecols, args.chunksize), start=1):
        checkpoint = checkpoint_dir / f"cells_{index:05d}.csv"
        record_path = record_dir / f"records_{index:05d}.csv.gz"
        checkpoint_paths.append(checkpoint)
        record_paths.append(record_path)
        if checkpoint.exists() and record_path.exists() and not args.force:
            print(f"pass 3 chunk {index}: checkpoints exist", flush=True)
            continue
        base = extract_base(chunk, mapping, args.dataset, end)
        features = text_features(base["text"])
        keep = keep_dates(base, start, end, args.min_words, features)
        base = base.loc[keep].copy()
        features = features.loc[keep]
        scored = apply_scores(features, params)
        frame = pd.concat([base.drop(columns=["text", "opened"]), scored], axis=1)
        opened = base["opened"]
        frame["post_chatgpt"] = opened.ge(cutoff).astype(int)
        frame["month"] = opened.dt.to_period("M").astype(str)
        frame["month_of_year"] = opened.dt.month.astype(int)
        frame["time_index"] = (opened.dt.year - start.year) * 12 + opened.dt.month - start.month
        frame["top_quartile_actionability"] = frame["actionability"].ge(high_threshold).astype(float)
        for name in keep_levels:
            frame[name] = collapse_levels(frame[name], keep_levels[name])
        frame = add_acs(frame, acs)
        record_columns = [
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
            *qcols,
            *numeric_cols,
        ]
        frame[record_columns].to_csv(record_path, index=False, compression="gzip")
        cells = aggregate_chunk(frame, group_cols, numeric_cols)
        cells.to_csv(checkpoint, index=False)
        print(f"pass 3 chunk {index}: {len(frame):,} records -> {len(cells):,} cells", flush=True)

    cells = combine_checkpoints(checkpoint_paths, group_cols, numeric_cols)
    cells.to_csv(outdir / f"{args.dataset}_corrected_cells.csv", index=False)
    monthly_summary(cells, numeric_cols).to_csv(outdir / f"{args.dataset}_corrected_monthly.csv", index=False)
    (outdir / f"{args.dataset}_analytic_record_manifest.txt").write_text(
        "\n".join(str(path) for path in record_paths) + "\n"
    )
    print(f"Done: {outdir}", flush=True)


if __name__ == "__main__":
    main()
