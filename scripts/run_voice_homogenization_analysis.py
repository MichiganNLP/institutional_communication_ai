#!/usr/bin/env python3
"""Research Line 4: within-topic voice diversity and template convergence."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from research_extension_common import ensure_directory, load_frame


WORD = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")
FIRST_PERSON = re.compile(r"\b(?:I|me|my|mine|we|our|ours)\b", re.I)
AFFECT = re.compile(r"\b(?:afraid|angry|anxious|concerned|distress|frustrat|hardship|stress|urgent|worried)\w*\b", re.I)


def normalized_template(text):
    value = str(text).lower()
    value = re.sub(r"\b\d+(?:[.,:/-]\d+)*\b", "<num>", value)
    value = re.sub(r"\s+", " ", value).strip()
    return " ".join(value.split()[:30])


def lexical_metrics(texts):
    tokens = [token.lower() for text in texts for token in WORD.findall(str(text))]
    if not tokens:
        return np.nan, np.nan
    counts = Counter(tokens)
    mattr_window = 50
    if len(tokens) <= mattr_window:
        mattr = len(counts) / len(tokens)
    else:
        starts = np.linspace(0, len(tokens) - mattr_window, min(500, len(tokens) - mattr_window + 1), dtype=int)
        mattr = np.mean([len(set(tokens[start : start + mattr_window])) / mattr_window for start in starts])
    return mattr, sum(1 for count in counts.values() if count == 1) / len(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", required=True, help="topic_assignments.csv.gz from Research Line 3")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max-per-cell", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_directory(outdir)
    data = load_frame(args.topics)
    rows = []
    rng = np.random.default_rng(args.seed)
    for (topic, post), group in data.groupby(["topic", "post_chatgpt"]):
        if len(group) > args.max_per_cell:
            group = group.iloc[rng.choice(len(group), args.max_per_cell, replace=False)]
        texts = group["text"].fillna("").astype(str).tolist()
        if len(texts) < 10:
            continue
        vectorizer = TfidfVectorizer(min_df=2, max_features=10000, ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        centroid = np.asarray(matrix.mean(axis=0))
        similarity = cosine_similarity(matrix, centroid).ravel()
        templates = pd.Series([normalized_template(text) for text in texts])
        duplicate_share = templates.duplicated(keep=False).mean()
        top_template_share = templates.value_counts(normalize=True).iloc[0]
        mattr, hapax_share = lexical_metrics(texts)
        words = np.maximum(group["text"].fillna("").str.count(WORD).to_numpy(float), 1.0)
        rows.append(
            {
                "dataset": args.dataset,
                "topic": topic,
                "post_chatgpt": int(post),
                "n": len(group),
                "mean_centroid_similarity": similarity.mean(),
                "mean_centroid_distance": 1.0 - similarity.mean(),
                "template_duplicate_share": duplicate_share,
                "top_template_share": top_template_share,
                "moving_average_type_token_ratio": mattr,
                "hapax_type_share": hapax_share,
                "first_person_per_100_words": 100.0 * group["text"].fillna("").str.count(FIRST_PERSON).sum() / words.sum(),
                "affect_per_100_words": 100.0 * group["text"].fillna("").str.count(AFFECT).sum() / words.sum(),
            }
        )
    cells = pd.DataFrame(rows)
    cells.to_csv(outdir / "voice_homogenization_topic_cells.csv", index=False)
    metrics = [column for column in cells if column not in {"dataset", "topic", "post_chatgpt", "n"}]
    changes = []
    for metric in metrics:
        pivot = cells.pivot(index="topic", columns="post_chatgpt", values=metric).dropna()
        for topic, row in pivot.iterrows():
            changes.append(
                {
                    "dataset": args.dataset,
                    "topic": topic,
                    "metric": metric,
                    "pre": row.get(0, np.nan),
                    "post": row.get(1, np.nan),
                    "post_minus_pre": row.get(1, np.nan) - row.get(0, np.nan),
                }
            )
    changes = pd.DataFrame(changes)
    changes.to_csv(outdir / "voice_homogenization_changes.csv", index=False)
    changes.groupby(["dataset", "metric"], as_index=False)["post_minus_pre"].agg(["mean", "median", "count"]).reset_index().to_csv(
        outdir / "voice_homogenization_summary.csv", index=False
    )
    print(f"Wrote Research Line 4 results to {outdir}", flush=True)


if __name__ == "__main__":
    main()
