#!/usr/bin/env python3
"""Research Line 5: institution profiles for information and presentation returns."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from absorbed_linear import fit_absorbed_ols
from research_extension_common import PRIMARY_OUTCOME, ensure_directory, load_frame, standardized_surface_score


def coefficient(result, name):
    table = result.table().set_index("term")
    if name not in table.index:
        return np.nan, np.nan, np.nan
    row = table.loc[name]
    return float(row["coef"]), float(row["se"]), float(row["pvalue"])


def profile(info_coef, info_p, polish_coef, polish_p, alpha):
    information = np.isfinite(info_p) and info_p < alpha and info_coef > 0
    presentation = np.isfinite(polish_p) and polish_p < alpha and polish_coef > 0
    if information and presentation:
        return "broadly_responsive"
    if information:
        return "evidence_responsive"
    if presentation:
        return "register_responsive"
    return "communication_insensitive_or_imprecise"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(PRIMARY_OUTCOME))
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--min-institution-n", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    outcome = PRIMARY_OUTCOME.get(args.dataset)
    if not outcome:
        raise ValueError(f"{args.dataset} has no record-level response outcome")
    outdir = Path(args.outdir)
    ensure_directory(outdir)
    data = load_frame(args.sample)
    _, data["surface_score"] = standardized_surface_score(data["text"].fillna(""))
    data["information_score"] = pd.to_numeric(data["actionability"], errors="coerce")
    data["information_post"] = data["information_score"] * data["post_chatgpt"]
    data["surface_post"] = data["surface_score"] * data["post_chatgpt"]
    rows = []
    for institution, group in data.groupby("institution", dropna=False):
        group = group.dropna(subset=[outcome, "information_score", "surface_score", "log_words"]).copy()
        if len(group) < args.min_institution_n or group[outcome].nunique() < 2:
            continue
        fixed_effects = [
            column
            for column in ["category", "subcategory", "month", "context", "state"]
            if column in group and group[column].nunique(dropna=False) > 1
        ]
        try:
            average = fit_absorbed_ols(
                group,
                outcome,
                ["information_score", "surface_score", "log_words"],
                fixed_effects,
                cluster="month",
            )
            temporal = fit_absorbed_ols(
                group,
                outcome,
                ["information_score", "information_post", "surface_score", "surface_post", "log_words"],
                fixed_effects,
                cluster="month",
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        info, info_se, info_p = coefficient(average, "information_score")
        polish, polish_se, polish_p = coefficient(average, "surface_score")
        info_post, info_post_se, info_post_p = coefficient(temporal, "information_post")
        polish_post, polish_post_se, polish_post_p = coefficient(temporal, "surface_post")
        rows.append(
            {
                "dataset": args.dataset,
                "institution": institution,
                "outcome": outcome,
                "n": len(group),
                "outcome_mean": group[outcome].mean(),
                "information_return": info,
                "information_se": info_se,
                "information_pvalue": info_p,
                "surface_return": polish,
                "surface_se": polish_se,
                "surface_pvalue": polish_p,
                "information_post_change": info_post,
                "information_post_change_se": info_post_se,
                "information_post_change_pvalue": info_post_p,
                "surface_post_change": polish_post,
                "surface_post_change_se": polish_post_se,
                "surface_post_change_pvalue": polish_post_p,
                "profile": profile(info, info_p, polish, polish_p, args.alpha),
                "fixed_effects": "+".join(fixed_effects),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "institution_conversion_profiles.csv", index=False)
    if len(result):
        result.groupby("profile", as_index=False).agg(
            institutions=("institution", "size"),
            records=("n", "sum"),
            mean_information_return=("information_return", "mean"),
            mean_surface_return=("surface_return", "mean"),
        ).to_csv(outdir / "institution_profile_summary.csv", index=False)
    print(f"Wrote Research Line 5 results to {outdir}", flush=True)


if __name__ == "__main__":
    main()

