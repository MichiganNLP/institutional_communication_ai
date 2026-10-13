#!/usr/bin/env python3
"""Research Line 2: rhetorical moves, missing information, and response returns."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from absorbed_linear import fit_absorbed_ols
from research_extension_common import MOVE_NAMES, PRIMARY_OUTCOME, add_rule_moves, ensure_directory, load_frame, save_frame


def model_moves(data, dataset, outdir):
    outcome = PRIMARY_OUTCOME.get(dataset)
    if not outcome or outcome not in data or data[outcome].dropna().nunique() < 2:
        return
    fixed_effects = [
        column
        for column in ["category", "subcategory", "month", "institution", "context", "state"]
        if column in data and data[column].nunique(dropna=False) > 1
    ]
    x = [f"move_{name}" for name in MOVE_NAMES]
    x += ["log_words"] if "log_words" in data else []
    for column in x:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    result = fit_absorbed_ols(data, outcome, x, fixed_effects, cluster="institution")
    table = result.table()
    table["dataset"] = dataset
    table["outcome"] = outcome
    table["n"] = result.n
    table["fixed_effects"] = "+".join(fixed_effects)
    table.to_csv(outdir / "rhetorical_move_response_returns.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(PRIMARY_OUTCOME))
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_directory(outdir)
    data = add_rule_moves(load_frame(args.sample))
    save_frame(data, outdir / "rhetorical_move_labels.csv.gz")

    rows = []
    for post, group in data.groupby("post_chatgpt"):
        for name in MOVE_NAMES:
            values = group[f"move_{name}"]
            rows.append(
                {
                    "dataset": args.dataset,
                    "post_chatgpt": int(post),
                    "move": name,
                    "n": len(values),
                    "prevalence": values.mean(),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "rhetorical_move_prevalence.csv", index=False)
    wide = summary.pivot(index="move", columns="post_chatgpt", values="prevalence").reset_index()
    if 0 in wide and 1 in wide:
        wide["post_minus_pre"] = wide[1] - wide[0]
    wide.insert(0, "dataset", args.dataset)
    wide.to_csv(outdir / "rhetorical_move_changes.csv", index=False)
    model_moves(data, args.dataset, outdir)
    print(f"Wrote Research Line 2 results to {outdir}", flush=True)


if __name__ == "__main__":
    main()
