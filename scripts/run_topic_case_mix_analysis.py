#!/usr/bin/env python3
"""Research Line 3: semantic topics and case-mix decomposition."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from research_extension_common import PRIMARY_OUTCOME, ensure_directory, load_frame, save_frame


def topic_terms(vectorizer, centers, top_n):
    terms = np.asarray(vectorizer.get_feature_names_out())
    result = {}
    for index, center in enumerate(centers):
        best = np.argsort(center)[-top_n:][::-1]
        result[index] = ", ".join(terms[best])
    return result


def decomposition(data, score):
    table = data.groupby(["topic", "post_chatgpt"])[score].agg(["mean", "size"]).reset_index()
    topics = sorted(data["topic"].unique())
    pre = data[data["post_chatgpt"].eq(0)]
    post = data[data["post_chatgpt"].eq(1)]
    p0 = pre["topic"].value_counts(normalize=True).reindex(topics, fill_value=0.0)
    p1 = post["topic"].value_counts(normalize=True).reindex(topics, fill_value=0.0)
    mu0 = pre.groupby("topic")[score].mean().reindex(topics)
    mu1 = post.groupby("topic")[score].mean().reindex(topics)
    pooled = data.groupby("topic")[score].mean().reindex(topics)
    mu0 = mu0.fillna(pooled)
    mu1 = mu1.fillna(pooled)
    observed = post[score].mean() - pre[score].mean()
    composition = float(((p1 - p0) * mu0).sum())
    within = float((p1 * (mu1 - mu0)).sum())
    return {
        "score": score,
        "pre_mean": pre[score].mean(),
        "post_mean": post[score].mean(),
        "observed_change": observed,
        "composition_effect": composition,
        "within_topic_effect": within,
        "reconstruction_error": observed - composition - within,
    }, table


def response_by_topic(data, dataset):
    outcome = PRIMARY_OUTCOME.get(dataset)
    if not outcome or outcome not in data or data[outcome].dropna().nunique() < 2:
        return pd.DataFrame()
    rows = []
    for topic, group in data.dropna(subset=[outcome, "actionability"]).groupby("topic"):
        if len(group) < 100 or group[outcome].nunique() < 2 or group["actionability"].std() < 1e-8:
            continue
        x = group["actionability"].to_numpy(float)
        y = group[outcome].to_numpy(float)
        coefficient = np.cov(x, y, ddof=0)[0, 1] / np.var(x)
        rows.append(
            {
                "dataset": dataset,
                "topic": topic,
                "n": len(group),
                "outcome": outcome,
                "outcome_mean": y.mean(),
                "information_return_unadjusted": coefficient,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(PRIMARY_OUTCOME))
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--topics", type=int, default=20)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--top-terms", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_directory(outdir)
    data = load_frame(args.sample)
    text = data["text"].fillna("").astype(str)
    vectorizer = TfidfVectorizer(
        min_df=5,
        max_df=0.95,
        max_features=args.max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(text)
    n_topics = min(args.topics, max(2, len(data) // 100))
    model = MiniBatchKMeans(
        n_clusters=n_topics,
        random_state=args.seed,
        batch_size=2048,
        n_init=10,
    )
    data["topic"] = model.fit_predict(matrix)
    labels = topic_terms(vectorizer, model.cluster_centers_, args.top_terms)
    data["topic_terms"] = data["topic"].map(labels)
    save_frame(data, outdir / "topic_assignments.csv.gz")

    topic_rows = []
    for topic, group in data.groupby("topic"):
        topic_rows.append(
            {
                "dataset": args.dataset,
                "topic": topic,
                "topic_terms": labels[topic],
                "n": len(group),
                "pre_n": int(group["post_chatgpt"].eq(0).sum()),
                "post_n": int(group["post_chatgpt"].eq(1).sum()),
                "prevalence": len(group) / len(data),
            }
        )
    pd.DataFrame(topic_rows).to_csv(outdir / "topic_summary.csv", index=False)

    decomposition_rows = []
    cell_tables = []
    for score in [column for column in ["actionability", "log_words"] if column in data]:
        row, cells = decomposition(data, score)
        row["dataset"] = args.dataset
        decomposition_rows.append(row)
        cells["dataset"] = args.dataset
        cells["score"] = score
        cell_tables.append(cells)
    pd.DataFrame(decomposition_rows).to_csv(outdir / "case_mix_decomposition.csv", index=False)
    pd.concat(cell_tables, ignore_index=True).to_csv(outdir / "topic_period_score_cells.csv", index=False)
    response_by_topic(data, args.dataset).to_csv(outdir / "topic_response_returns.csv", index=False)
    print(f"Wrote Research Line 3 results to {outdir}", flush=True)


if __name__ == "__main__":
    main()

