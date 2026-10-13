#!/usr/bin/env python3
"""Merge CFPB narratives with ACS ZCTA context and summarize demographic gaps.

This is a descriptive first pass and does not fit fixed-effect regression models.
It asks whether high-legibility complaint-writing
returns vary across neighborhood-context quartiles.
"""

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CHATGPT_DATE = pd.Timestamp("2022-11-30")
RELIEF = {"Closed with monetary relief", "Closed with non-monetary relief"}
VALID_RESPONSES = {
    "Closed with monetary relief",
    "Closed with non-monetary relief",
    "Closed with explanation",
    "Closed",
    "Untimely response",
    "Closed with administrative response",
}

LEGAL_PATTERN = r"\b(pursuant|violation|violated|regulation|section|statute|fcra|fdcpa|evidence|documentation|investigate|investigation|dispute|disputed|furnisher|furnishers|credit reporting act|debt collection practices)\b"
STRUCTURE_PATTERN = r"\b(account number|account name|reason|date|amount|reference|case number|confirmation|attached|documentation|please remove|request that)\b"
EMOTION_PATTERN = r"\b(angry|upset|frustrated|stress|stressed|anxiety|anxious|scared|devastated|terrible|horrible|unfair|ridiculous)\b"

RENAME = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Product": "product",
    "Issue": "issue",
    "Sub-product": "sub_product",
    "Sub-issue": "sub_issue",
    "Company": "company",
    "Company response to consumer": "company_response",
    "Timely response?": "timely",
    "Submitted via": "submitted_via",
    "State": "state",
    "ZIP code": "zip_code",
    "Tags": "tags",
    "Consumer complaint narrative": "complaint_what_happened",
}

FEATURE_COLS = ["log_words", "legal_term_count", "structure_term_count", "newline_count", "emotion_term_count", "redaction_rate"]
ACS_VARS = [
    "median_household_income",
    "poverty_rate",
    "unemployment_rate",
    "bachelors_plus_share",
    "nh_white_share",
    "nh_black_share",
    "hispanic_share",
    "foreign_born_share",
    "limited_english_household_share",
    "broadband_share",
    "homeownership_share",
    "renter_share",
    "age65_plus_share",
]


def normalize_columns(df):
    return df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})


def prepare_chunk(df):
    df = normalize_columns(df).copy()
    for col in ["complaint_id", "date_received", "product", "issue", "company", "company_response", "state", "zip_code", "complaint_what_happened"]:
        if col not in df.columns:
            df[col] = pd.NA

    text = df["complaint_what_happened"].fillna("").astype(str)
    dates = pd.to_datetime(df["date_received"], errors="coerce", utc=True).dt.tz_localize(None)
    zip_code = df["zip_code"].fillna("").astype(str).str.strip()
    word_count = text.str.split().str.len().fillna(0).astype(float)

    out = pd.DataFrame(index=df.index)
    out["complaint_id"] = df["complaint_id"]
    out["date_received"] = dates
    out["post_chatgpt"] = (dates >= CHATGPT_DATE).astype(int)
    out["product"] = df["product"].fillna("Missing")
    out["company"] = df["company"].fillna("Missing")
    out["state"] = df["state"].fillna("Missing")
    out["zip_code"] = zip_code
    out["zcta"] = zip_code.where(zip_code.str.match(r"^\d{5}$", na=False))
    out["company_response"] = df["company_response"].fillna("Missing")
    out["valid_response"] = out["company_response"].isin(VALID_RESPONSES)
    out["valid_core_fields"] = out["product"].ne("Missing") & out["company"].ne("Missing") & out["state"].ne("Missing") & out["date_received"].notna()
    out["base_sample"] = out["valid_response"] & out["valid_core_fields"] & out["zcta"].notna()
    out["relief"] = out["company_response"].isin(RELIEF).astype(int)
    out["word_count"] = word_count
    out["text_len_500plus"] = text.str.len() >= 500
    out["log_words"] = np.log(word_count + 1.0)
    out["legal_term_count"] = text.str.count(LEGAL_PATTERN, flags=re.I).astype(float)
    out["structure_term_count"] = text.str.count(STRUCTURE_PATTERN, flags=re.I).astype(float)
    out["newline_count"] = text.str.count(r"\n").astype(float)
    out["emotion_term_count"] = text.str.count(EMOTION_PATTERN, flags=re.I).astype(float)
    out["redaction_rate"] = text.str.count("XXXX").astype(float) / word_count.clip(lower=1.0)
    return out


def update_moments(stats, frame):
    eligible = frame[frame["base_sample"]]
    stats["n"] += len(eligible)
    for col in FEATURE_COLS:
        vals = eligible[col].astype(float)
        stats[col + "_sum"] += float(vals.sum())
        stats[col + "_sumsq"] += float((vals * vals).sum())


def finalize_moments(stats):
    params = {}
    n = max(stats["n"], 1)
    for col in FEATURE_COLS:
        mean = stats[col + "_sum"] / n
        var = max(stats[col + "_sumsq"] / n - mean * mean, 0.0)
        params[col] = (mean, math.sqrt(var) if var > 0 else 1.0)
    return params


def add_legibility_score(frame, params):
    score = np.zeros(len(frame), dtype=float)
    score += (frame["log_words"] - params["log_words"][0]) / params["log_words"][1]
    score += (frame["legal_term_count"] - params["legal_term_count"][0]) / params["legal_term_count"][1]
    score += (frame["structure_term_count"] - params["structure_term_count"][0]) / params["structure_term_count"][1]
    score += (frame["newline_count"] - params["newline_count"][0]) / params["newline_count"][1]
    score -= 0.5 * ((frame["emotion_term_count"] - params["emotion_term_count"][0]) / params["emotion_term_count"][1])
    score -= 0.25 * ((frame["redaction_rate"] - params["redaction_rate"][0]) / params["redaction_rate"][1])
    frame["legibility_score"] = score
    return frame


def load_acs(path):
    acs = pd.read_csv(path, dtype={"zcta": str})
    acs["zcta"] = acs["zcta"].astype(str).str.zfill(5)
    keep = ["zcta"] + [v for v in ACS_VARS if v in acs.columns]
    return acs[keep]


def acs_quartile_maps(acs):
    maps = {}
    for var in ACS_VARS:
        if var not in acs.columns:
            continue
        vals = pd.to_numeric(acs[var], errors="coerce")
        valid = vals.dropna()
        if valid.nunique() < 4:
            continue
        try:
            codes = pd.qcut(vals, 4, labels=False, duplicates="drop")
        except ValueError:
            continue
        max_code = codes.dropna().max()
        if pd.isna(max_code):
            continue
        max_code = int(max_code)
        if max_code == 3:
            labels = {0: "Q1_low", 1: "Q2", 2: "Q3", 3: "Q4_high"}
        else:
            labels = {i: "Q{}".format(i + 1) for i in range(max_code + 1)}
            labels[0] = "Q1_low"
            labels[max_code] = "Q{}_high".format(max_code + 1)
        q = codes.map(labels)
        maps[var] = dict(zip(acs["zcta"], q))
    return maps


def payoff_table(df, group_col, min_n):
    g = df.groupby([group_col, "high_legibility"], dropna=False).agg(n=("complaint_id", "count"), relief_sum=("relief", "sum"), high_legibility_share=("high_legibility", "mean")).reset_index()
    wide_n = g.pivot(index=group_col, columns="high_legibility", values="n")
    wide_r = g.pivot(index=group_col, columns="high_legibility", values="relief_sum")
    out = pd.DataFrame(index=wide_n.index)
    out["n_0"] = wide_n.get(0, 0)
    out["n_1"] = wide_n.get(1, 0)
    out["relief_rate_0"] = wide_r.get(0, 0) / out["n_0"].replace(0, np.nan)
    out["relief_rate_1"] = wide_r.get(1, 0) / out["n_1"].replace(0, np.nan)
    out = out.reset_index()
    out = out[(out["n_0"].fillna(0) >= min_n) & (out["n_1"].fillna(0) >= min_n)].copy()
    out["legibility_return_pp"] = 100 * (out["relief_rate_1"] - out["relief_rate_0"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfpb", required=True)
    parser.add_argument("--acs", required=True)
    parser.add_argument("--outdir", default="outputs/acs_gap_experiments_2021_2024")
    parser.add_argument("--chunksize", type=int, default=100000)
    parser.add_argument("--min-group-n", type=int, default=1000)
    parser.add_argument("--require-text-500", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    acs = load_acs(args.acs)
    acs_maps = acs_quartile_maps(acs)

    stats = defaultdict(float)
    print("pass 1: feature moments for clean-ZIP ACS universe")
    for i, chunk in enumerate(pd.read_csv(args.cfpb, dtype=str, chunksize=args.chunksize), start=1):
        frame = prepare_chunk(chunk)
        if args.require_text_500:
            frame["base_sample"] = frame["base_sample"] & frame["text_len_500plus"]
        update_moments(stats, frame)
        print("pass 1 chunk {}".format(i))
    params = finalize_moments(stats)

    scores = []
    print("pass 2: threshold")
    for i, chunk in enumerate(pd.read_csv(args.cfpb, dtype=str, chunksize=args.chunksize), start=1):
        frame = add_legibility_score(prepare_chunk(chunk), params)
        if args.require_text_500:
            frame["base_sample"] = frame["base_sample"] & frame["text_len_500plus"]
        scores.append(frame.loc[frame["base_sample"], "legibility_score"].to_numpy())
        print("pass 2 chunk {}".format(i))
    threshold = float(np.quantile(np.concatenate(scores), 0.75))

    parts = []
    adoption_parts = []
    print("pass 3: merge ACS and aggregate")
    for i, chunk in enumerate(pd.read_csv(args.cfpb, dtype=str, chunksize=args.chunksize), start=1):
        frame = add_legibility_score(prepare_chunk(chunk), params)
        if args.require_text_500:
            frame["base_sample"] = frame["base_sample"] & frame["text_len_500plus"]
        frame = frame[frame["base_sample"]].copy()
        frame["high_legibility"] = (frame["legibility_score"] >= threshold).astype(int)
        frame = frame.merge(acs, on="zcta", how="left")
        frame = frame[frame[ACS_VARS].notna().any(axis=1)].copy()
        for var, qmap in acs_maps.items():
            frame[var + "_quartile"] = frame["zcta"].map(qmap)
            small = frame[["complaint_id", "relief", "high_legibility", "post_chatgpt", var + "_quartile"]].dropna()
            small = small.rename(columns={var + "_quartile": "quartile"})
            small["acs_var"] = var
            parts.append(small[["acs_var", "quartile", "complaint_id", "relief", "high_legibility"]])
            adoption_parts.append(small.groupby(["acs_var", "quartile", "post_chatgpt"], dropna=False).agg(n=("complaint_id", "count"), high_legibility_sum=("high_legibility", "sum"), relief_sum=("relief", "sum")).reset_index())
        print("pass 3 chunk {}".format(i))

    all_returns = pd.concat(parts, ignore_index=True)
    returns = []
    for var, df_var in all_returns.groupby("acs_var"):
        tab = payoff_table(df_var, "quartile", args.min_group_n)
        tab.insert(0, "acs_var", var)
        returns.append(tab)
    returns = pd.concat(returns, ignore_index=True)
    returns.to_csv(outdir / "acs_legibility_returns_by_quartile.csv", index=False)

    adoption = pd.concat(adoption_parts, ignore_index=True)
    adoption = adoption.groupby(["acs_var", "quartile", "post_chatgpt"], dropna=False).agg(n=("n", "sum"), high_legibility_sum=("high_legibility_sum", "sum"), relief_sum=("relief_sum", "sum")).reset_index()
    adoption["high_legibility_share"] = adoption["high_legibility_sum"] / adoption["n"]
    adoption["relief_rate"] = adoption["relief_sum"] / adoption["n"]
    adoption.to_csv(outdir / "acs_prepost_by_quartile.csv", index=False)

    pd.DataFrame([{"metric": "base_sample_n", "value": int(stats["n"])}, {"metric": "high_legibility_threshold", "value": threshold}]).to_csv(outdir / "acs_thresholds.csv", index=False)
    print("wrote", outdir)


if __name__ == "__main__":
    main()
