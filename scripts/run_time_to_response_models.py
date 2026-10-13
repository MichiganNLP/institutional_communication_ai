#!/usr/bin/env python3
"""Censored Cox response-time models for San Diego and FOIA."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg


def load_records(paneldir, dataset):
    manifest = Path(paneldir) / f"{dataset}_analytic_record_manifest.txt"
    parts = [part for path in manifest.read_text().splitlines() if path and len(part := pd.read_csv(path, low_memory=False))]
    if not parts:
        raise ValueError("No analytic records found")
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["sandiego", "foia"])
    parser.add_argument("--paneldir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--covariance", choices=["auto", "model_based"], default="auto")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_records(args.paneldir, args.dataset)
    data = data.dropna(subset=["response_duration_hours", "response_event", "actionability", "log_words"]).copy()
    data = data[data["response_duration_hours"].gt(0)]
    if args.max_records > 0 and len(data) > args.max_records:
        data = data.sample(args.max_records, random_state=2026)
    data["duration_days"] = data["response_duration_hours"] / 24.0
    data["actionability_post"] = data["actionability"] * data["post_chatgpt"]
    month_dummies = pd.get_dummies(data["month_of_year"].astype(str), prefix="month", drop_first=True, dtype=float)
    exog = pd.concat(
        [
            data[["actionability", "actionability_post", "post_chatgpt", "time_index", "log_words"]].astype(float).reset_index(drop=True),
            month_dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    strata = data["institution"].fillna("Missing").astype(str) + "|" + data["category"].fillna("Missing").astype(str)
    groups = data["institution"].fillna("Missing").astype(str)
    model = PHReg(
        data["duration_days"].to_numpy(),
        exog.to_numpy(),
        status=data["response_event"].astype(int).to_numpy(),
        strata=strata.to_numpy(),
        ties="efron",
    )
    cluster_count = groups.nunique(dropna=False)
    if args.covariance == "model_based":
        result = model.fit()
        covariance = "model_based_requested"
    elif cluster_count >= 10:
        result = model.fit(groups=groups.to_numpy())
        covariance = "institution_clustered"
        if not np.isfinite(result.bse).all():
            result = model.fit()
            covariance = "model_based_cluster_covariance_nonfinite"
    else:
        result = model.fit()
        covariance = "model_based_fewer_than_10_institutions"
    table = pd.DataFrame(
        {
            "term": exog.columns,
            "log_hazard_coef": result.params,
            "se": result.bse,
            "hazard_ratio": np.exp(result.params),
            "ci_low": np.exp(result.params - 1.96 * result.bse),
            "ci_high": np.exp(result.params + 1.96 * result.bse),
            "pvalue": result.pvalues,
            "n": len(data),
            "events": int(data["response_event"].sum()),
            "dataset": args.dataset,
            "covariance": covariance,
            "institution_clusters": cluster_count,
        }
    )
    table.to_csv(outdir / "time_to_response_cox.csv", index=False)
    print(f"Done: {outdir}", flush=True)


if __name__ == "__main__":
    main()
