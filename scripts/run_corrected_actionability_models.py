#!/usr/bin/env python3
"""Run corrected temporal, response-return, neighborhood, and institution models."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from absorbed_linear import fit_absorbed_ols
from corrected_actionability_common import ACS_VARS, COMPONENTS, SCORE_COLUMNS, ensure_directory, load_acs_quartiles


DEFAULT_OUTCOMES = {
    "cfpb": ["response_favorable", "response_timely"],
    "nhtsa": [],
    "sandiego": ["response_timely"],
    "foia": ["response_timely", "response_favorable", "response_complete"],
}
PRIMARY_OUTCOME = {
    "cfpb": "response_favorable",
    "sandiego": "response_timely",
    "foia": "response_favorable",
}
CUTOFFS = {
    "actual_2022_11": "2022-11-30",
    "alternative_2023_01": "2023-01-31",
    "alternative_2023_03": "2023-03-31",
    "placebo_2021_11": "2021-11-30",
    "placebo_2022_05": "2022-05-31",
}


def load_records(paneldir, dataset, max_records=None, usecols=None):
    manifest = paneldir / f"{dataset}_analytic_record_manifest.txt"
    paths = [Path(line) for line in manifest.read_text().splitlines() if line.strip()]
    parts = []
    remaining = max_records
    for path in paths:
        part = pd.read_csv(
            path,
            usecols=(lambda column: column in usecols) if usecols is not None else None,
            low_memory=False,
        )
        if remaining is not None:
            part = part.head(remaining)
            remaining -= len(part)
        if len(part):
            parts.append(part)
        if remaining is not None and remaining <= 0:
            break
    if not parts:
        raise ValueError("No analytic records found in the manifest")
    result = pd.concat(parts, ignore_index=True)
    for column in ["category", "subcategory", "institution", "context", "state", "month"]:
        if column in result:
            result[column] = result[column].fillna("Missing").astype("category")
    return result


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


def period_columns(frame, cutoff):
    cutoff_month = pd.Timestamp(cutoff).to_period("M") + 1
    month_period = pd.PeriodIndex(frame["month"], freq="M")
    cutoff_index = (cutoff_month.year - 2021) * 12 + cutoff_month.month - 1
    result = frame.copy()
    result["model_post"] = (month_period >= cutoff_month).astype(float)
    result["model_time_after"] = np.maximum(pd.to_numeric(result["time_index"]) - cutoff_index, 0.0)
    return result


def run_temporal(cells, dataset, outdir):
    output = []
    temporal_fes = ["category", "institution", "month_of_year"]
    temporal_metrics = ["actionability", *COMPONENTS, *[c for c in SCORE_COLUMNS if c != "actionability"]]
    for cutoff_name, cutoff in CUTOFFS.items():
        data = period_columns(cells, cutoff)
        metrics = temporal_metrics if cutoff_name == "actual_2022_11" else ["actionability"]
        for metric in metrics:
            usable = data.dropna(subset=[metric]).copy()
            for exclusion in ([0, 1, 3] if cutoff_name == "actual_2022_11" and metric == "actionability" else [0]):
                if exclusion:
                    cutoff_period = pd.Timestamp(cutoff).to_period("M") + 1
                    distance = np.abs(usable["month"].map(lambda value: pd.Period(value, freq="M").ordinal - cutoff_period.ordinal))
                    model_data = usable[distance > exclusion].copy()
                else:
                    model_data = usable
                result = fit_absorbed_ols(
                    model_data,
                    metric,
                    ["time_index", "model_post", "model_time_after"],
                    temporal_fes,
                    weight="n",
                    cluster="month",
                )
                metadata = {
                    "dataset": dataset,
                    "analysis": "interrupted_time_series",
                    "metric": metric,
                    "cutoff": cutoff,
                    "cutoff_name": cutoff_name,
                    "excluded_months_each_side": exclusion,
                    "fixed_effects": "+".join(temporal_fes),
                }
                output.append(write_result(result, outdir / f"temporal_{cutoff_name}_{metric}_exclude{exclusion}.csv", metadata))

    balanced = period_columns(cells, CUTOFFS["actual_2022_11"])
    institution_periods = balanced.groupby("institution")["model_post"].nunique()
    category_periods = balanced.groupby("category")["model_post"].nunique()
    balanced = balanced[
        balanced["institution"].isin(institution_periods[institution_periods.eq(2)].index)
        & balanced["category"].isin(category_periods[category_periods.eq(2)].index)
    ].copy()
    if len(balanced) >= 100:
        result = fit_absorbed_ols(
            balanced,
            "actionability",
            ["time_index", "model_post", "model_time_after"],
            temporal_fes,
            weight="n",
            cluster="month",
        )
        metadata = {
            "dataset": dataset,
            "analysis": "interrupted_time_series_balanced_levels",
            "metric": "actionability",
            "cutoff": CUTOFFS["actual_2022_11"],
            "cutoff_name": "actual_2022_11_balanced",
            "excluded_months_each_side": 0,
            "fixed_effects": "+".join(temporal_fes),
        }
        output.append(write_result(result, outdir / "temporal_actual_2022_11_actionability_balanced.csv", metadata))
    pd.concat(output, ignore_index=True).to_csv(outdir / "temporal_model_index.csv", index=False)


def run_response(records, dataset, outcomes, outdir):
    output = []
    marginal_rows = []
    fixed_effects = ["category", "subcategory", "month", "institution", "context"]
    records = records.copy()
    records["actionability_post"] = records["actionability"] * records["post_chatgpt"]
    records["log_words_post"] = records["log_words"] * records["post_chatgpt"]
    for outcome in outcomes:
        if outcome not in records or records[outcome].notna().sum() < 100:
            continue
        result = fit_absorbed_ols(
            records,
            outcome,
            ["actionability", "actionability_post", "log_words", "log_words_post"],
            fixed_effects,
            cluster="institution",
        )
        metadata = {
            "dataset": dataset,
            "analysis": "response_return",
            "outcome": outcome,
            "score": "actionability",
            "fixed_effects": "+".join(fixed_effects),
        }
        output.append(write_result(result, outdir / f"response_{outcome}.csv", metadata))
        for post in [0, 1]:
            estimate, se, pvalue = result.contrast(
                {"actionability": 1.0, "actionability_post": float(post)}
            )
            marginal_rows.append(
                {
                    "dataset": dataset,
                    "outcome": outcome,
                    "post_chatgpt": post,
                    "marginal_actionability_slope": estimate,
                    "se": se,
                    "pvalue": pvalue,
                    "n": result.n,
                }
            )

    main_outcome = PRIMARY_OUTCOME.get(dataset)
    if main_outcome and main_outcome in records:
        for score in [column for column in SCORE_COLUMNS if column != "actionability"]:
            data = records.copy()
            interaction = score + "_post"
            data[interaction] = data[score] * data["post_chatgpt"]
            result = fit_absorbed_ols(data, main_outcome, [score, interaction], fixed_effects, cluster="institution")
            metadata = {
                "dataset": dataset,
                "analysis": "response_score_robustness",
                "outcome": main_outcome,
                "score": score,
                "fixed_effects": "+".join(fixed_effects),
            }
            output.append(write_result(result, outdir / f"response_robustness_{score}.csv", metadata))
    if output:
        pd.concat(output, ignore_index=True).to_csv(outdir / "response_model_index.csv", index=False)
    if marginal_rows:
        pd.DataFrame(marginal_rows).to_csv(outdir / "response_marginal_returns.csv", index=False)


def neighborhood_design(data, qcolumn):
    result = data
    levels = [level for level in ["Q1_low", "Q2", "Q3", "Q4_high"] if (result[qcolumn] == level).any()]
    if len(levels) < 2 or "Q1_low" not in levels:
        return None, None
    result["actionability_post"] = result["actionability"] * result["post_chatgpt"]
    result["log_words_post"] = result["log_words"] * result["post_chatgpt"]
    x = ["actionability", "actionability_post", "log_words", "log_words_post"]
    for level in levels[1:]:
        suffix = level.lower()
        result[f"q_{suffix}"] = (result[qcolumn] == level).astype(float)
        result[f"q_{suffix}_post"] = result[f"q_{suffix}"] * result["post_chatgpt"]
        result[f"actionability_q_{suffix}"] = result["actionability"] * result[f"q_{suffix}"]
        result[f"actionability_post_q_{suffix}"] = result["actionability_post"] * result[f"q_{suffix}"]
        x.extend([f"q_{suffix}", f"q_{suffix}_post", f"actionability_q_{suffix}", f"actionability_post_q_{suffix}"])
    return result, (levels, x)


def run_neighborhood(records, dataset, outcomes, outdir):
    qcolumns = [variable + "_q" for variable in ACS_VARS if variable + "_q" in records]
    if not qcolumns:
        return
    fixed_effects = ["category", "subcategory", "month", "institution", "context"]
    coefficient_tables = []
    marginal_rows = []
    production_rows = []
    primary = PRIMARY_OUTCOME.get(dataset)
    for qcolumn in qcolumns:
        return_path = outdir / f"neighborhood_return_{qcolumn}.csv"
        production_path = outdir / f"neighborhood_production_{qcolumn}.csv"
        marginal_path = outdir / f"neighborhood_marginal_returns_{qcolumn}.csv"
        if production_path.is_file() and (primary not in outcomes or return_path.is_file()) and marginal_path.is_file():
            print(f"{dataset} {qcolumn}: already complete", flush=True)
            if return_path.is_file():
                coefficient_tables.append(pd.read_csv(return_path))
            production_rows.append(pd.read_csv(production_path))
            marginal_rows.extend(pd.read_csv(marginal_path).to_dict("records"))
            continue

        needed = {
            qcolumn, "zcta", "actionability", "post_chatgpt", "log_words",
            "time_index", "month", "month_of_year", "category", "subcategory",
            "institution", "context", primary,
        }
        columns = [column for column in needed if column and column in records]
        source = records.loc[records[qcolumn].notna(), columns].copy()
        print(f"{dataset} {qcolumn}: n={len(source):,}", flush=True)
        designed, description = neighborhood_design(source, qcolumn)
        if description is None:
            continue
        levels, x = description
        if primary in outcomes and primary in designed and designed[primary].notna().sum() >= 100:
            result = fit_absorbed_ols(designed, primary, x, fixed_effects, cluster="institution")
            metadata = {
                "dataset": dataset,
                "analysis": "neighborhood_return",
                "outcome": primary,
                "acs_variable": qcolumn[:-2],
                "fixed_effects": "+".join(fixed_effects),
            }
            coefficient_tables.append(write_result(result, return_path, metadata))
            q_marginal_rows = []
            for level in levels:
                suffix = level.lower()
                for post in [0, 1]:
                    weights = {"actionability": 1.0}
                    if post:
                        weights["actionability_post"] = 1.0
                    if level != "Q1_low":
                        weights[f"actionability_q_{suffix}"] = 1.0
                        if post:
                            weights[f"actionability_post_q_{suffix}"] = 1.0
                    estimate, se, pvalue = result.contrast(weights)
                    cell_mask = designed[qcolumn].eq(level) & designed["post_chatgpt"].eq(post)
                    row = {
                            "dataset": dataset,
                            "outcome": primary,
                            "acs_variable": qcolumn[:-2],
                            "quartile": level,
                            "post_chatgpt": post,
                            "marginal_actionability_slope": estimate,
                            "se": se,
                            "pvalue": pvalue,
                            "n": result.n,
                            "cell_n": int(cell_mask.sum()),
                            "cell_zcta_n": int(designed.loc[cell_mask, "zcta"].nunique()) if "zcta" in designed else np.nan,
                        }
                    marginal_rows.append(row)
                    q_marginal_rows.append(row)
            pd.DataFrame(q_marginal_rows).to_csv(marginal_path, index=False)
            del result
        elif not marginal_path.is_file():
            pd.DataFrame().to_csv(marginal_path, index=False)

        # Production-gap interrupted trend by quartile.
        production_columns = [
            column for column in [
                qcolumn, "actionability", "time_index", "month", "month_of_year",
                "category", "institution",
            ] if column in designed
        ]
        production = period_columns(designed[production_columns], CUTOFFS["actual_2022_11"])
        del designed, source
        gc.collect()
        levels = [level for level in ["Q1_low", "Q2", "Q3", "Q4_high"] if (production[qcolumn] == level).any()]
        xprod = ["time_index", "model_post", "model_time_after"]
        for level in levels[1:]:
            suffix = level.lower()
            production[f"q_{suffix}"] = (production[qcolumn] == level).astype(float)
            production[f"time_q_{suffix}"] = production["time_index"] * production[f"q_{suffix}"]
            production[f"post_q_{suffix}"] = production["model_post"] * production[f"q_{suffix}"]
            production[f"time_after_q_{suffix}"] = production["model_time_after"] * production[f"q_{suffix}"]
            xprod.extend([f"q_{suffix}", f"time_q_{suffix}", f"post_q_{suffix}", f"time_after_q_{suffix}"])
        result = fit_absorbed_ols(
            production,
            "actionability",
            xprod,
            ["category", "institution", "month_of_year"],
            cluster="month",
        )
        table = result.table()
        table["dataset"] = dataset
        table["acs_variable"] = qcolumn[:-2]
        table["analysis"] = "neighborhood_production"
        table["n"] = result.n
        table.to_csv(production_path, index=False)
        production_rows.append(table)
        del result, production, table
        gc.collect()

    if coefficient_tables:
        pd.concat(coefficient_tables, ignore_index=True).to_csv(outdir / "neighborhood_return_model_index.csv", index=False)
    if marginal_rows:
        pd.DataFrame(marginal_rows).to_csv(outdir / "neighborhood_marginal_returns.csv", index=False)
    if production_rows:
        pd.concat(production_rows, ignore_index=True).to_csv(outdir / "neighborhood_production_model_index.csv", index=False)


def run_institution(records, dataset, outcomes, outdir, minimum):
    primary = PRIMARY_OUTCOME.get(dataset)
    if not primary or primary not in outcomes:
        return
    rows = []
    for institution, group in records.groupby("institution", observed=True):
        usable_n = int(group[primary].notna().sum())
        if usable_n < minimum or group["post_chatgpt"].nunique() < 2:
            continue
        data = group.copy()
        data["actionability_post"] = data["actionability"] * data["post_chatgpt"]
        data["log_words_post"] = data["log_words"] * data["post_chatgpt"]
        try:
            result = fit_absorbed_ols(
                data,
                primary,
                ["actionability", "actionability_post", "log_words", "log_words_post"],
                ["category", "subcategory", "month", "context"],
                cluster="month",
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        for post in [0, 1]:
            weights = {"actionability": 1.0, "actionability_post": float(post)}
            estimate, se, pvalue = result.contrast(weights)
            rows.append(
                {
                    "dataset": dataset,
                    "institution": str(institution),
                    "outcome": primary,
                    "post_chatgpt": post,
                    "marginal_actionability_slope": estimate,
                    "se": se,
                    "pvalue": pvalue,
                    "n": result.n,
                }
            )
    result_frame = pd.DataFrame(rows)
    if len(result_frame):
        result_frame["shrunken_marginal_slope"] = np.nan
        result_frame["shrunken_se"] = np.nan
        for post, index in result_frame.groupby("post_chatgpt").groups.items():
            group = result_frame.loc[index]
            variance = group["se"].pow(2).clip(lower=1e-12)
            fixed_weights = 1.0 / variance
            slopes = group["marginal_actionability_slope"]
            fixed_center = float(np.average(slopes, weights=fixed_weights))
            q_stat = float((fixed_weights * slopes.sub(fixed_center).pow(2)).sum())
            c_stat = float(fixed_weights.sum() - fixed_weights.pow(2).sum() / fixed_weights.sum())
            between = max((q_stat - (len(group) - 1)) / c_stat, 0.0) if len(group) > 1 and c_stat > 0 else 0.0
            if between > 0:
                random_weights = 1.0 / (variance + between)
                center = float(np.average(slopes, weights=random_weights))
                precision = 1.0 / variance + 1.0 / between
                posterior = (slopes / variance + center / between) / precision
                posterior_se = np.sqrt(1.0 / precision)
            else:
                center = fixed_center
                pooled_se = float(np.sqrt(1.0 / fixed_weights.sum()))
                posterior = pd.Series(center, index=group.index)
                posterior_se = pd.Series(pooled_se, index=group.index)
            result_frame.loc[index, "shrunken_marginal_slope"] = posterior
            result_frame.loc[index, "shrunken_se"] = posterior_se
    result_frame.to_csv(outdir / "institution_marginal_returns.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DEFAULT_OUTCOMES))
    parser.add_argument("--paneldir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--acs", default=None, help="Optional ACS file used to add missing ZIP-level quartile columns")
    parser.add_argument(
        "--sample-specific-acs-quartiles",
        action="store_true",
        help="Rebuild ACS quartiles over unique ZCTAs observed in this dataset and replace existing national quartiles",
    )
    parser.add_argument("--analyses", default="temporal,response,neighborhood,institution")
    parser.add_argument("--outcomes", nargs="*")
    parser.add_argument("--min-institution-records", type=int, default=5000)
    parser.add_argument("--max-records", type=int, help="Testing only: cap loaded analytic records")
    args = parser.parse_args()

    paneldir = Path(args.paneldir)
    outdir = Path(args.outdir)
    ensure_directory(outdir)
    analyses = {value.strip() for value in args.analyses.split(",") if value.strip()}
    outcomes = args.outcomes if args.outcomes is not None else DEFAULT_OUTCOMES[args.dataset]

    cells = None
    records = None
    if "temporal" in analyses:
        cells = pd.read_csv(paneldir / f"{args.dataset}_corrected_cells.csv", low_memory=False)
        run_temporal(cells, args.dataset, outdir)
    if analyses & {"response", "neighborhood", "institution"}:
        record_columns = {
            "zip", "zcta", "category", "subcategory", "month", "month_of_year",
            "institution", "context", "time_index", "post_chatgpt", "log_words",
            *SCORE_COLUMNS, *outcomes, *[variable + "_q" for variable in ACS_VARS],
        }
        records = load_records(paneldir, args.dataset, args.max_records, usecols=record_columns)
        if args.acs and "zip" in records:
            records["zcta"] = records["zip"].astype(str).str.extract(r"(\d{5})", expand=False)
            observed_zctas = records["zcta"] if args.sample_specific_acs_quartiles else None
            acs = load_acs_quartiles(args.acs, zctas=observed_zctas)
            acs_qcols = [column for column in acs.columns if column != "zcta"]
            if args.sample_specific_acs_quartiles:
                records = records.drop(columns=[column for column in acs_qcols if column in records])
                merge_qcols = acs_qcols
            else:
                merge_qcols = [column for column in acs_qcols if column not in records]
            if merge_qcols:
                records = records.merge(acs[["zcta", *merge_qcols]], on="zcta", how="left", validate="many_to_one")
    if "response" in analyses and outcomes:
        run_response(records, args.dataset, outcomes, outdir)
    if "neighborhood" in analyses and outcomes:
        run_neighborhood(records, args.dataset, outcomes, outdir)
    if "institution" in analyses and outcomes:
        run_institution(records, args.dataset, outcomes, outdir, args.min_institution_records)

    (outdir / "run_metadata.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "paneldir": str(paneldir),
                "analyses": sorted(analyses),
                "outcomes": outcomes,
                "max_records": args.max_records,
                "sample_specific_acs_quartiles": args.sample_specific_acs_quartiles,
            },
            indent=2,
        )
    )
    print(f"Done: {outdir}", flush=True)


if __name__ == "__main__":
    main()
