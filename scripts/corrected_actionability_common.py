#!/usr/bin/env python3
"""Shared schema, NLP feature, outcome, and weighted-model helpers."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


CHATGPT_DATE = pd.Timestamp("2022-11-30")
COMPONENTS = ["specificity", "chronology", "evidence", "procedural", "remedy"]
SCORE_COLUMNS = [
    "actionability",
    "actionability_with_length",
    "length_only",
    *[f"actionability_without_{c}" for c in COMPONENTS],
]
ACS_VARS = [
    "median_household_income",
    "poverty_rate",
    "unemployment_rate",
    "bachelors_plus_share",
    "limited_english_household_share",
    "broadband_share",
    "homeownership_share",
    "renter_share",
]


DATASETS = {
    "cfpb": {
        "id": ["Complaint ID"],
        "date": ["Date received"],
        "text": ["Consumer complaint narrative"],
        "category": ["Product"],
        "subcategory": ["Issue", "Sub-issue"],
        "institution": ["Company"],
        "context": ["Submitted via"],
        "state": ["State"],
        "zip": ["ZIP code"],
        "closed": [],
        "status": ["Company response to consumer"],
        "timely": ["Timely response?"],
    },
    "nhtsa": {
        "id": ["odi_number", "cmplid"],
        "date": ["date_complaint_filed_parsed", "date_complaint_filed"],
        "text": ["summary"],
        "category": ["component"],
        "subcategory": ["product_type", "model"],
        "institution": ["manufacturer", "make"],
        "context": ["state", "make"],
        "state": ["state"],
        "zip": [],
        "closed": [],
        "status": [],
        "timely": [],
    },
    "sandiego": {
        "id": ["service_request_id"],
        "date": ["date_requested", "requested_datetime"],
        "text": ["public_description", "description"],
        "category": ["service_name"],
        "subcategory": ["service_name_detail", "case_record_type"],
        "institution": ["case_record_type", "agency_responsible", "department"],
        "context": ["comm_plan_name", "council_district"],
        "state": [],
        "zip": ["zipcode", "zip_code"],
        "closed": ["date_closed", "closed_datetime"],
        "status": ["status"],
        "timely": [],
    },
    "foia": {
        "id": ["request_id", "request_number", "tracking_number"],
        "date": ["original_received_date", "received_date", "original_date_received", "perfected_date"],
        "text": ["request_description", "request_detail", "description"],
        "category": ["subject_category", "request_type", "request_default_category"],
        "subcategory": ["multi_track_type", "request_track", "requester_default_category"],
        "institution": ["foia_source_agency", "action_office"],
        "context": ["foia_source_agency", "action_office"],
        "state": [],
        "zip": [],
        "closed": ["closed_date", "original_closed_date", "date_request_closed"],
        "status": ["final_disposition", "final_dispositions", "disposition_details", "request_status", "status"],
        "timely": [],
    },
}


TERM_PATTERNS = {
    "chronology": re.compile(
        r"\b(after|before|then|when|since|during|again|first|next|finally|previously|"
        r"subsequently|later|earlier|until|while|following|prior to)\b",
        re.I,
    ),
    "evidence": re.compile(
        r"\b(document|documentation|attached|attachment|evidence|photo|photograph|video|"
        r"receipt|invoice|record|screenshot|proof|copy|statement|letter|email|report)\w*\b",
        re.I,
    ),
    "procedural": re.compile(
        r"\b(called|emailed|submitted|reported|contacted|notified|filed|appealed|disputed|"
        r"case|ticket|claim|reference|account|representative|supervisor|department|agency|"
        r"investigation|complaint|confirmation)\w*\b",
        re.I,
    ),
    "remedy": re.compile(
        r"\b(request|refund|repair|replace|resolve|resolution|investigate|correct|remove|"
        r"respond|restore|reimburse|cancel|close|fix|explain|provide|release|waive|stop)\w*\b",
        re.I,
    ),
}

DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|(?:19|20)\d{2}|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.I,
)
MONEY_PATTERN = re.compile(r"(?:\$\s?\d[\d,]*(?:\.\d{1,2})?|\b\d[\d,]*(?:\.\d{1,2})?\s?(?:dollars?|usd)\b)", re.I)
IDENTIFIER_PATTERN = re.compile(
    r"\b(?:case|ticket|claim|account|reference|confirmation|invoice|order|report|vin)"
    r"(?:\s+(?:number|no\.?))?\s*[:#-]?\s*[a-z0-9-]{3,}\b",
    re.I,
)
NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")
WORD_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")


def norm(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def resolve_columns(header, spec):
    lower = {str(c).lower(): c for c in header}
    return {
        key: [lower[c.lower()] for c in candidates if c.lower() in lower]
        for key, candidates in spec.items()
    }


def coalesce(frame: pd.DataFrame, columns) -> pd.Series:
    if not columns:
        return pd.Series("", index=frame.index, dtype=object)
    result = norm(frame[columns[0]])
    for column in columns[1:]:
        candidate = norm(frame[column])
        result = result.where(result.ne(""), candidate)
    return result


def parse_dates(frame: pd.DataFrame, columns) -> pd.Series:
    raw = coalesce(frame, columns)
    try:
        parsed = pd.to_datetime(raw, format="mixed", errors="coerce", utc=True)
    except (TypeError, ValueError):
        parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def text_features(text: pd.Series) -> pd.DataFrame:
    text = norm(text)
    words = text.str.count(WORD_PATTERN).astype(float)
    denominator = words.clip(lower=20.0)

    chronology_count = text.str.count(TERM_PATTERNS["chronology"]).astype(float)
    evidence_count = text.str.count(TERM_PATTERNS["evidence"]).astype(float)
    procedural_count = text.str.count(TERM_PATTERNS["procedural"]).astype(float)
    remedy_count = text.str.count(TERM_PATTERNS["remedy"]).astype(float)
    specificity_count = (
        text.str.count(DATE_PATTERN).astype(float)
        + text.str.count(MONEY_PATTERN).astype(float)
        + text.str.count(IDENTIFIER_PATTERN).astype(float)
        + 0.25 * text.str.count(NUMBER_PATTERN).astype(float)
    )

    result = pd.DataFrame(index=text.index)
    result["word_count"] = words
    result["log_words"] = np.log1p(words)
    result["specificity"] = np.log1p(100.0 * specificity_count / denominator)
    result["chronology"] = np.log1p(100.0 * chronology_count / denominator)
    result["evidence"] = np.log1p(100.0 * evidence_count / denominator)
    result["procedural"] = np.log1p(100.0 * procedural_count / denominator)
    result["remedy"] = np.log1p(100.0 * remedy_count / denominator)
    return result


def fit_reference_parameters(feature_arrays):
    params = {}
    for feature, arrays in feature_arrays.items():
        values = np.concatenate(arrays).astype(float) if arrays else np.array([0.0])
        values = values[np.isfinite(values)]
        cap = float(np.quantile(values, 0.99)) if len(values) else 0.0
        clipped = np.minimum(values, cap)
        mean = float(clipped.mean()) if len(clipped) else 0.0
        sd = float(clipped.std(ddof=0)) if len(clipped) else 1.0
        params[feature] = {"cap": cap, "mean": mean, "sd": max(sd, 1e-8)}
    return params


def apply_scores(features: pd.DataFrame, params) -> pd.DataFrame:
    z = pd.DataFrame(index=features.index)
    for feature in [*COMPONENTS, "log_words"]:
        p = params[feature]
        values = features[feature].clip(upper=p["cap"])
        z[feature] = (values - p["mean"]) / p["sd"]

    result = features.copy()
    result["actionability"] = z[COMPONENTS].mean(axis=1)
    result["actionability_with_length"] = z[[*COMPONENTS, "log_words"]].mean(axis=1)
    result["length_only"] = z["log_words"]
    for omitted in COMPONENTS:
        result[f"actionability_without_{omitted}"] = z[[c for c in COMPONENTS if c != omitted]].mean(axis=1)
    return result


def extract_base(frame, mapping, dataset, end_date):
    opened = parse_dates(frame, mapping["date"])
    text = coalesce(frame, mapping["text"])
    closed = parse_dates(frame, mapping["closed"])
    status = coalesce(frame, mapping["status"])

    result = pd.DataFrame(index=frame.index)
    result["record_id"] = coalesce(frame, mapping["id"])
    result["opened"] = opened
    result["text"] = text
    for key in ["category", "subcategory", "institution", "context", "state"]:
        value = coalesce(frame, mapping[key])
        result[key] = value.mask(value.eq(""), "Missing")
    result["zip"] = coalesce(frame, mapping["zip"]).str.extract(r"(\d{5})", expand=False)
    result["response_favorable"] = np.nan
    result["response_timely"] = np.nan
    result["response_complete"] = np.nan
    result["response_hours"] = np.nan
    result["response_event"] = np.nan
    result["response_duration_hours"] = np.nan

    if dataset == "cfpb":
        known = status.ne("")
        result.loc[known, "response_favorable"] = status[known].isin(
            ["Closed with monetary relief", "Closed with non-monetary relief"]
        ).astype(float)
        timely = coalesce(frame, mapping["timely"]).str.upper()
        result.loc[timely.isin(["YES", "NO"]), "response_timely"] = timely[timely.isin(["YES", "NO"])].eq("YES").astype(float)
        result.loc[known, "response_complete"] = 1.0

    elif dataset in {"sandiego", "foia"}:
        hours = (closed - opened).dt.total_seconds() / 3600.0
        result["response_hours"] = hours.where(hours.ge(0))
        event = closed.notna() & closed.ge(opened) & closed.lt(end_date)
        censor_or_close = closed.where(event, end_date)
        duration = (censor_or_close - opened).dt.total_seconds() / 3600.0
        result["response_event"] = event.astype(float)
        result["response_duration_hours"] = duration.where(duration.ge(0))
        final_status = status.str.upper()
        final_pattern = r"CLOSED|COMPLETE|COMPLETED|RESOLVED|GRANT|DENI|NO RECORD|WITHDRAW|REJECT"
        complete = closed.notna() | final_status.str.contains(final_pattern, regex=True, na=False)
        result.loc[complete, "response_complete"] = 1.0

        deadline_days = 7 if dataset == "sandiego" else 60
        observed_window = opened.le(end_date - pd.Timedelta(days=deadline_days))
        result.loc[observed_window & ~complete, "response_complete"] = 0.0
        timely = hours.ge(0) & hours.le(deadline_days * 24)
        result.loc[observed_window, "response_timely"] = timely[observed_window].astype(float)

        if dataset == "sandiego":
            result["response_favorable"] = result["response_timely"]
        else:
            positive = final_status.str.contains(r"GRANT|PARTIAL", regex=True, na=False)
            negative = final_status.str.contains(r"DENI|NO RECORD|WITHDRAW|REJECT|CLOSED", regex=True, na=False)
            result.loc[positive, "response_favorable"] = 1.0
            result.loc[negative & ~positive, "response_favorable"] = 0.0

    return result


def load_acs_quartiles(path, zctas=None):
    if not path:
        return None
    acs = pd.read_csv(path, dtype={"zcta": str})
    acs["zcta"] = acs["zcta"].astype(str).str.zfill(5)
    if zctas is not None:
        observed = pd.Series(list(zctas), dtype="string").str.zfill(5).dropna().unique()
        acs = acs[acs["zcta"].isin(observed)].copy()
    keep = ["zcta"]
    for variable in ACS_VARS:
        if variable not in acs.columns:
            continue
        numeric = pd.to_numeric(acs[variable], errors="coerce")
        try:
            acs[variable + "_q"] = pd.qcut(
                numeric,
                4,
                labels=["Q1_low", "Q2", "Q3", "Q4_high"],
                duplicates="drop",
            ).astype(object)
            keep.append(variable + "_q")
        except ValueError:
            # For zero-inflated measures, keep the tied minimum together and
            # split the remaining positive values into three equal groups.
            grouped = pd.Series(np.nan, index=acs.index, dtype=object)
            minimum = numeric.min(skipna=True)
            minimum_mask = numeric.eq(minimum)
            grouped.loc[minimum_mask] = "Q1_low"
            remainder = numeric.loc[numeric.notna() & ~minimum_mask]
            grouped.loc[remainder.index] = pd.qcut(
                remainder,
                3,
                labels=["Q2", "Q3", "Q4_high"],
                duplicates="drop",
            ).astype(object)
            acs[variable + "_q"] = grouped
            if acs[variable + "_q"].nunique(dropna=True) >= 2:
                keep.append(variable + "_q")
    return acs[keep]


def add_acs(frame, acs):
    if acs is None or "zip" not in frame:
        return frame
    return frame.merge(acs, left_on="zip", right_on="zcta", how="left").drop(columns=["zcta"])


def weighted_mean(frame, value, weight="n"):
    usable = frame[value].notna() & frame[weight].gt(0)
    if not usable.any():
        return np.nan
    return float(np.average(frame.loc[usable, value], weights=frame.loc[usable, weight]))


def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)
