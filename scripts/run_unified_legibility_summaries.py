#!/usr/bin/env python3
"""Recompute descriptive prevalence and pooled returns with one legibility score."""

import argparse
from pathlib import Path

import pandas as pd

from absorbed_linear import fit_absorbed_ols


DATASETS = ["cfpb", "nhtsa", "sandiego", "foia"]
OUTCOMES = {
    "cfpb": "response_favorable",
    "sandiego": "response_timely",
    "foia": "response_favorable",
}
LABELS = {
    "cfpb": "CFPB",
    "nhtsa": "NHTSA",
    "sandiego": "San Diego",
    "foia": "FOIA",
}
FIXED_EFFECTS = ["category", "subcategory", "month", "institution", "context"]


def load_cells(root: Path, dataset: str):
    directory = root / dataset / "panel" / "_cell_checkpoints"
    paths = sorted(directory.glob("cells_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No cell checkpoints in {directory}")
    outcome = OUTCOMES.get(dataset)
    columns = [
        "post_chatgpt",
        *FIXED_EFFECTS,
        "n",
        "actionability_sum",
        "actionability_n",
        "top_quartile_actionability_sum",
        "top_quartile_actionability_n",
        "log_words_sum",
        "log_words_n",
    ]
    if outcome:
        columns.extend([f"{outcome}_sum", f"{outcome}_n"])
    parts = [pd.read_csv(path, usecols=columns, low_memory=False) for path in paths]
    frame = pd.concat(parts, ignore_index=True)
    frame["legibility"] = frame["actionability_sum"] / frame["actionability_n"].replace(0, pd.NA)
    frame["high_legibility"] = (
        frame["top_quartile_actionability_sum"]
        / frame["top_quartile_actionability_n"].replace(0, pd.NA)
    )
    frame["log_words"] = frame["log_words_sum"] / frame["log_words_n"].replace(0, pd.NA)
    if outcome:
        frame["response"] = frame[f"{outcome}_sum"] / frame[f"{outcome}_n"].replace(0, pd.NA)
        frame["response_weight"] = frame[f"{outcome}_n"]
    return frame


def descriptive_rows(frame, dataset):
    rows = []
    for post, period in [(0, "Pre"), (1, "Post")]:
        subset = frame[frame["post_chatgpt"].eq(post)]
        high_denominator = subset["top_quartile_actionability_n"].sum()
        score_denominator = subset["actionability_n"].sum()
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": LABELS[dataset],
                "period": period,
                "n": int(subset["n"].sum()),
                "high_legibility_share": (
                    subset["top_quartile_actionability_sum"].sum() / high_denominator
                ),
                "mean_legibility": subset["actionability_sum"].sum() / score_denominator,
            }
        )
    return rows


def pooled_return_rows(frame, dataset):
    if dataset not in OUTCOMES:
        return []
    data = frame.dropna(subset=["response", "legibility", "log_words"]).copy()
    rows = []
    specifications = {
        "fixed_effects": ["legibility"],
        "fixed_effects_plus_length": ["legibility", "log_words"],
    }
    for name, predictors in specifications.items():
        result = fit_absorbed_ols(
            data,
            "response",
            predictors,
            FIXED_EFFECTS,
            weight="response_weight",
            cluster="institution",
        )
        table = result.table().set_index("term")
        row = table.loc["legibility"]
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": LABELS[dataset],
                "outcome": OUTCOMES[dataset],
                "specification": name,
                "coefficient": row["coef"],
                "se": row["se"],
                "pvalue": row["pvalue"],
                "n_cells": result.n,
                "weighted_n": int(data["response_weight"].sum()),
                "cluster_count": result.cluster_count,
                "r2_within": result.r2_within,
            }
        )
    return rows


def load_records(root: Path, dataset: str):
    panel = root / dataset / "panel"
    manifest = panel / f"{dataset}_analytic_record_manifest.txt"
    paths = [Path(line) for line in manifest.read_text().splitlines() if line.strip()]
    outcome = OUTCOMES[dataset]
    columns = [
        *FIXED_EFFECTS,
        "actionability",
        "top_quartile_actionability",
        "log_words",
        outcome,
    ]
    parts = [pd.read_csv(path, usecols=columns, low_memory=False) for path in paths]
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.rename(
        columns={
            "actionability": "legibility",
            "top_quartile_actionability": "high_legibility",
            outcome: "response",
        }
    )
    return frame


def record_return_rows(frame, dataset):
    data = frame.dropna(subset=["response", "legibility", "log_words"]).copy()
    rows = []
    specifications = {
        "continuous_fixed_effects": ("legibility", ["legibility"]),
        "continuous_fixed_effects_plus_length": ("legibility", ["legibility", "log_words"]),
        "high_legibility_fixed_effects_plus_length": (
            "high_legibility",
            ["high_legibility", "log_words"],
        ),
    }
    for name, (target, predictors) in specifications.items():
        result = fit_absorbed_ols(
            data,
            "response",
            predictors,
            FIXED_EFFECTS,
            cluster="institution",
        )
        row = result.table().set_index("term").loc[target]
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": LABELS[dataset],
                "outcome": OUTCOMES[dataset],
                "specification": name,
                "score_term": target,
                "coefficient": row["coef"],
                "se": row["se"],
                "pvalue": row["pvalue"],
                "n_records": result.n,
                "cluster_count": result.cluster_count,
                "r2_within": result.r2_within,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    descriptive = []
    returns = []
    record_returns = []
    for dataset in args.datasets:
        print(f"Loading {dataset} corrected analytic cells", flush=True)
        frame = load_cells(root, dataset)
        descriptive.extend(descriptive_rows(frame, dataset))
        returns.extend(pooled_return_rows(frame, dataset))
        print(f"Finished {dataset}: {len(frame):,} analytic cells", flush=True)
        if dataset in OUTCOMES:
            print(f"Loading {dataset} corrected analytic records", flush=True)
            records = load_records(root, dataset)
            record_returns.extend(record_return_rows(records, dataset))
            print(f"Finished {dataset}: {len(records):,} analytic records", flush=True)

    pd.DataFrame(descriptive).to_csv(outdir / "unified_legibility_prepost.csv", index=False)
    pd.DataFrame(returns).to_csv(outdir / "unified_legibility_average_returns.csv", index=False)
    pd.DataFrame(record_returns).to_csv(
        outdir / "unified_legibility_record_average_returns.csv", index=False
    )
    print(f"Wrote unified summaries to {outdir}", flush=True)


if __name__ == "__main__":
    main()
