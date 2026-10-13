#!/usr/bin/env python3
"""Run a transparent specification grid for legibility-response associations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from absorbed_linear import fit_absorbed_ols
from corrected_actionability_common import SCORE_COLUMNS
from run_corrected_actionability_models import DEFAULT_OUTCOMES, PRIMARY_OUTCOME, load_records


def existing_fes(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in frame and frame[column].nunique(dropna=False) > 1]


def prepare(frame: pd.DataFrame, score: str, include_length: bool) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    interaction = score + "_post_sensitivity"
    result[interaction] = pd.to_numeric(result[score], errors="coerce") * pd.to_numeric(result["post_chatgpt"], errors="coerce")
    x = [score, interaction]
    if include_length:
        result["log_words_post_sensitivity"] = pd.to_numeric(result["log_words"], errors="coerce") * pd.to_numeric(result["post_chatgpt"], errors="coerce")
        x += ["log_words", "log_words_post_sensitivity"]
    return result, x


def focal_rows(result, dataset: str, outcome: str, specification: str, score: str, sample: str, fixed_effects: list[str], cluster: str) -> list[dict]:
    interaction = score + "_post_sensitivity"
    rows = []
    for period, weights in [
        ("pre", {score: 1.0}),
        ("post", {score: 1.0, interaction: 1.0}),
        ("post_minus_pre", {interaction: 1.0}),
    ]:
        estimate, se, pvalue = result.contrast(weights)
        rows.append(
            {
                "dataset": dataset, "outcome": outcome, "specification": specification,
                "score": score, "sample": sample, "estimand": period, "estimate": estimate,
                "se": se, "pvalue": pvalue, "n": result.n, "r2_within": result.r2_within,
                "fixed_effects": "+".join(fixed_effects), "cluster": cluster,
                "cluster_count": result.cluster_count,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DEFAULT_OUTCOMES))
    parser.add_argument("--paneldir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--outcomes", nargs="*")
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame = load_records(Path(args.paneldir), args.dataset, args.max_records)
    outcomes = args.outcomes if args.outcomes is not None else DEFAULT_OUTCOMES[args.dataset]
    outcomes = [outcome for outcome in outcomes if outcome in frame and frame[outcome].notna().sum() >= 100]
    if not outcomes:
        raise ValueError(f"No usable institutional response outcomes for {args.dataset}")

    full_fe = existing_fes(frame, ["category", "subcategory", "month", "institution", "context"])
    reduced_fe = existing_fes(frame, ["category", "month", "institution"])
    cluster_main = "institution" if "institution" in frame and frame["institution"].nunique() >= 10 else "month"
    specifications = [
        ("main", "actionability", True, full_fe, cluster_main, "all"),
        ("no_length_control", "actionability", False, full_fe, cluster_main, "all"),
        ("reduced_fixed_effects", "actionability", True, reduced_fe, cluster_main, "all"),
        ("month_clustered", "actionability", True, full_fe, "month", "all"),
        ("trim_score_1_99", "actionability", True, full_fe, cluster_main, "trim_score_1_99"),
        ("exclude_cutoff_3_months", "actionability", True, full_fe, cluster_main, "exclude_cutoff_3_months"),
    ]
    for score in SCORE_COLUMNS:
        if score != "actionability" and score in frame:
            specifications.append((f"alternate_score_{score}", score, True, full_fe, cluster_main, "all"))

    focal = []
    coefficients = []
    for outcome in outcomes:
        for name, score, include_length, fixed_effects, cluster, sample_name in specifications:
            data = frame
            if sample_name == "trim_score_1_99":
                low, high = pd.to_numeric(data[score], errors="coerce").quantile([0.01, 0.99])
                data = data[pd.to_numeric(data[score], errors="coerce").between(low, high)].copy()
            elif sample_name == "exclude_cutoff_3_months":
                month = pd.PeriodIndex(data["month"].astype(str), freq="M")
                cutoff = pd.Period("2022-12", freq="M")
                distance = np.abs(np.array([value.ordinal - cutoff.ordinal for value in month]))
                data = data[distance > 3].copy()
            model_data, x = prepare(data, score, include_length)
            try:
                result = fit_absorbed_ols(model_data, outcome, x, fixed_effects, cluster=cluster)
            except (ValueError, np.linalg.LinAlgError) as error:
                focal.append({"dataset": args.dataset, "outcome": outcome, "specification": name, "score": score, "sample": sample_name, "estimand": "ERROR", "error": str(error)})
                continue
            focal.extend(focal_rows(result, args.dataset, outcome, name, score, sample_name, fixed_effects, cluster))
            table = result.table()
            table.insert(0, "dataset", args.dataset)
            table.insert(1, "outcome", outcome)
            table.insert(2, "specification", name)
            table["score"] = score
            table["n"] = result.n
            coefficients.append(table)
            print(f"{args.dataset} {outcome} {name}: n={result.n:,}", flush=True)

    pd.DataFrame(focal).to_csv(outdir / "outcome_sensitivity_focal_estimates.csv", index=False)
    if coefficients:
        pd.concat(coefficients, ignore_index=True).to_csv(outdir / "outcome_sensitivity_all_coefficients.csv", index=False)
    (outdir / "run_metadata.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(f"Wrote sensitivity grid to {outdir}")


if __name__ == "__main__":
    main()
