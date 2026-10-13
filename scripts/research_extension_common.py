#!/usr/bin/env python3
"""Shared helpers for the additional institutional-communication research lines."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_OUTCOME = {
    "cfpb": "response_favorable",
    "nhtsa": None,
    "sandiego": "response_timely",
    "foia": "response_favorable",
}

MOVE_PATTERNS = {
    "problem_harm": re.compile(
        r"\b(?:charged|denied|damaged|failed|fraud|harm|incorrect|issue|lost|missing|problem|"
        r"refused|unsafe|violation|wrong|unable|unresolved|overcharged|delay)\w*\b",
        re.I,
    ),
    "actor_object": re.compile(
        r"\b(?:account|agency|bank|card|company|consumer|department|driver|loan|merchant|"
        r"property|record|request|road|service|vehicle|manufacturer|office)\w*\b",
        re.I,
    ),
    "chronology": re.compile(
        r"\b(?:after|before|then|when|since|during|again|first|next|finally|previously|"
        r"subsequently|later|earlier|until|while|following|prior to)\b",
        re.I,
    ),
    "evidence": re.compile(
        r"\b(?:document|documentation|attached|attachment|evidence|photo|photograph|video|"
        r"receipt|invoice|record|screenshot|proof|copy|statement|letter|email|report)\w*\b",
        re.I,
    ),
    "prior_contact": re.compile(
        r"\b(?:called|emailed|contacted|notified|reported|submitted|filed|appealed|disputed|"
        r"spoke|representative|supervisor|ticket|case|confirmation)\w*\b",
        re.I,
    ),
    "procedural_context": re.compile(
        r"\b(?:agency|appeal|case|claim|complaint|department|dispute|investigation|policy|"
        r"procedure|process|regulation|request|review|ticket)\w*\b",
        re.I,
    ),
    "requested_remedy": re.compile(
        r"\b(?:I request|please|refund|repair|replace|resolve|resolution|investigate|correct|"
        r"remove|respond|restore|reimburse|cancel|fix|explain|provide|release|waive|stop)\w*\b",
        re.I,
    ),
    "affect_consequence": re.compile(
        r"\b(?:afraid|angry|anxious|concerned|distress|frustrat|harm|hardship|humiliat|"
        r"stress|urgent|worried|cannot afford|unable to)\w*\b",
        re.I,
    ),
}

MOVE_NAMES = list(MOVE_PATTERNS)
NUMBER_TOKEN = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b|\$\s?\d[\d,.]*")
WORD_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")


def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_frame(path):
    path = Path(path)
    return pd.read_csv(path, low_memory=False, compression="gzip" if path.suffix == ".gz" else None)


def save_frame(frame, path):
    path = Path(path)
    ensure_directory(path.parent)
    frame.to_csv(path, index=False, compression="gzip" if path.suffix == ".gz" else None)


def add_rule_moves(frame, text_column="text"):
    result = frame.copy()
    text = result[text_column].fillna("").astype(str)
    for name, pattern in MOVE_PATTERNS.items():
        result[f"move_{name}"] = text.str.contains(pattern, regex=True, na=False).astype(np.int8)
    columns = [f"move_{name}" for name in MOVE_NAMES]
    result["move_count"] = result[columns].sum(axis=1)
    result["move_coverage"] = result["move_count"] / len(columns)
    return result


def deterministic_sample(frame, n, seed=2026, strata=None):
    if n <= 0 or len(frame) <= n:
        return frame.copy()
    data = frame.copy()
    key = data.get("record_id", data.index.to_series()).astype(str) + f"|{seed}"
    data["_priority"] = pd.util.hash_pandas_object(key, index=False).astype("uint64")
    if not strata:
        return data.nsmallest(n, "_priority").drop(columns="_priority")
    groups = data.groupby(strata, dropna=False)
    per_group = max(1, n // max(groups.ngroups, 1))
    selected = groups.apply(lambda part: part.nsmallest(per_group, "_priority"), include_groups=False)
    selected = selected.reset_index().drop(columns=[c for c in ["_priority", "level_0", "level_1"] if c in selected])
    if len(selected) < n:
        observed = set(selected.get("record_id", pd.Series(dtype=str)).astype(str))
        remainder = data[~data.get("record_id", data.index.to_series()).astype(str).isin(observed)]
        selected = pd.concat([selected, remainder.nsmallest(n - len(selected), "_priority").drop(columns="_priority")])
    return selected.head(n).reset_index(drop=True)


def standardized_surface_score(text):
    from run_information_polish_analysis import SURFACE_COMPONENTS, surface_features

    raw = surface_features(text)
    z = raw.copy()
    for column in SURFACE_COMPONENTS:
        sd = raw[column].std(ddof=0)
        z[column] = (raw[column] - raw[column].mean()) / (sd if sd > 1e-8 else 1.0)
    return raw, z[SURFACE_COMPONENTS].mean(axis=1)


def extract_numbers(text):
    return set(NUMBER_TOKEN.findall(str(text)))


def lexical_jaccard(left, right):
    a = {word.lower() for word in WORD_TOKEN.findall(str(left))}
    b = {word.lower() for word in WORD_TOKEN.findall(str(right))}
    return len(a & b) / len(a | b) if a or b else 1.0


def parse_json_object(text):
    value = str(text).strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

