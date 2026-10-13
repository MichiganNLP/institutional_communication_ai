#!/usr/bin/env python3
"""Topic-conditioned semantic convergence before and after public AI diffusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from research_extension_common import ensure_directory, load_frame


METRIC_DIRECTION = {
    "pairwise_cosine_similarity": 1,
    "period_centroid_similarity": 1,
    "semantic_dispersion": -1,
    "similarity_to_pre_centroid": 1,
}


def priority(frame: pd.DataFrame, seed: int) -> pd.Series:
    key = frame["record_id"].astype(str) + f"|{seed}"
    return pd.util.hash_pandas_object(key, index=False).astype("uint64")


def balanced_sample(data: pd.DataFrame, mode: str, cap: int, seed: int) -> pd.DataFrame:
    work = data.copy()
    work["length_bin"] = work.groupby("topic")["log_words"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop")
    )
    strata = ["topic", "length_bin"]
    if mode == "topic_category_length":
        work["category_match"] = work["category"].fillna("MISSING").astype(str)
        strata.append("category_match")
    work["_priority"] = priority(work, seed)
    selected = []
    for _, group in work.groupby(strata, dropna=False):
        counts = group["post_chatgpt"].value_counts()
        if 0 not in counts or 1 not in counts:
            continue
        n = min(int(counts[0]), int(counts[1]), cap)
        if n < 2:
            continue
        for post in (0, 1):
            selected.append(group[group["post_chatgpt"] == post].nsmallest(n, "_priority"))
    if not selected:
        raise ValueError(f"No matched pre/post strata for mode={mode}")
    return pd.concat(selected, ignore_index=True).drop(columns="_priority")


def pairwise_similarity(matrix: np.ndarray, rng: np.random.Generator, max_pairs: int) -> float:
    n = len(matrix)
    if n < 2:
        return np.nan
    total = n * (n - 1) // 2
    if total <= max_pairs and n <= 1500:
        gram = matrix @ matrix.T
        return float(gram[np.triu_indices(n, 1)].mean())
    left = rng.integers(0, n, size=max_pairs)
    right = rng.integers(0, n, size=max_pairs)
    keep = left != right
    return float(np.sum(matrix[left[keep]] * matrix[right[keep]], axis=1).mean())


def bootstrap_summary(changes: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for (dataset, mode, metric), group in changes.groupby(["dataset", "match_mode", "metric"]):
        values = group["post_minus_pre"].dropna().to_numpy(float)
        if not len(values):
            continue
        boot = np.mean(rng.choice(values, size=(draws, len(values)), replace=True), axis=1)
        estimate = float(values.mean())
        p = 2 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
        direction = METRIC_DIRECTION[metric]
        rows.append({
            "dataset": dataset,
            "match_mode": mode,
            "metric": metric,
            "post_minus_pre": estimate,
            "convergence_oriented_change": direction * estimate,
            "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
            "bootstrap_p": min(p, 1.0),
            "n_topics": len(values),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-per-stratum-period", type=int, default=250)
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_directory(outdir)
    data = load_frame(args.topics)
    required = {"record_id", "text", "topic", "post_chatgpt", "category", "log_words"}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    data = data.dropna(subset=["text", "topic", "post_chatgpt"]).copy()
    data["record_id"] = data["record_id"].astype(str)
    data["text"] = data["text"].astype(str)
    data = data[data["text"].str.split().str.len() >= 5]

    samples = []
    for offset, mode in enumerate(("topic_length", "topic_category_length")):
        sample = balanced_sample(data, mode, args.max_per_stratum_period, args.seed + offset)
        sample["match_mode"] = mode
        samples.append(sample)
    analysis = pd.concat(samples, ignore_index=True)
    unique = analysis[["record_id", "text"]].drop_duplicates("record_id").reset_index(drop=True)
    print(f"Encoding {len(unique):,} unique requests with {args.embedding_model}", flush=True)
    encoder = SentenceTransformer(args.embedding_model, device=args.device)
    embeddings = encoder.encode(
        unique["text"].tolist(), batch_size=args.batch_size, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    )
    vector_by_id = {record_id: embeddings[i] for i, record_id in enumerate(unique["record_id"])}

    rows = []
    rng = np.random.default_rng(args.seed)
    for (mode, topic), topic_group in analysis.groupby(["match_mode", "topic"]):
        pre_ids = topic_group.loc[topic_group["post_chatgpt"] == 0, "record_id"]
        if len(pre_ids) < 2:
            continue
        pre_matrix = np.vstack([vector_by_id[x] for x in pre_ids])
        pre_centroid = pre_matrix.mean(axis=0)
        pre_centroid /= max(np.linalg.norm(pre_centroid), 1e-12)
        for post, group in topic_group.groupby("post_chatgpt"):
            matrix = np.vstack([vector_by_id[x] for x in group["record_id"]])
            centroid = matrix.mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-12)
            similarity = matrix @ centroid
            rows.append({
                "dataset": args.dataset,
                "match_mode": mode,
                "topic": topic,
                "post_chatgpt": int(post),
                "n": len(group),
                "pairwise_cosine_similarity": pairwise_similarity(matrix, rng, args.max_pairs),
                "period_centroid_similarity": float(similarity.mean()),
                "semantic_dispersion": float((1.0 - similarity).mean()),
                "similarity_to_pre_centroid": float((matrix @ pre_centroid).mean()),
            })
    cells = pd.DataFrame(rows)
    cells.to_csv(outdir / "semantic_voice_topic_cells.csv", index=False)

    changes = []
    for (mode, topic), group in cells.groupby(["match_mode", "topic"]):
        for metric in METRIC_DIRECTION:
            values = group.set_index("post_chatgpt")[metric]
            if 0 in values and 1 in values:
                delta = float(values.loc[1] - values.loc[0])
                changes.append({
                    "dataset": args.dataset, "match_mode": mode, "topic": topic,
                    "metric": metric, "pre": float(values.loc[0]), "post": float(values.loc[1]),
                    "post_minus_pre": delta,
                    "convergence_oriented_change": METRIC_DIRECTION[metric] * delta,
                })
    changes = pd.DataFrame(changes)
    changes.to_csv(outdir / "semantic_voice_topic_changes.csv", index=False)
    summary = bootstrap_summary(changes, args.bootstrap, args.seed)
    summary.to_csv(outdir / "semantic_voice_dataset_summary.csv", index=False)
    pd.DataFrame({
        "embedding_model": [args.embedding_model], "matching_modes": ["topic_length;topic_category_length"],
        "interpretation": ["Positive convergence_oriented_change indicates greater post-cutoff homogenization."],
    }).to_csv(outdir / "semantic_voice_run_metadata.csv", index=False)
    print(f"Wrote semantic convergence results to {outdir}", flush=True)


if __name__ == "__main__":
    main()
