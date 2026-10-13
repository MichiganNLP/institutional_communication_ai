#!/usr/bin/env python3
"""Research Line 6: prepare and analyze fact-preserving register counterfactuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from corrected_actionability_common import COMPONENTS, text_features
from research_extension_common import (
    PRIMARY_OUTCOME,
    deterministic_sample,
    ensure_directory,
    extract_numbers,
    lexical_jaccard,
    load_frame,
)
from run_information_polish_analysis import SURFACE_COMPONENTS, surface_features


CONDITIONS = {
    "minimal_correction": "Correct only obvious grammar, spelling, and punctuation. Preserve wording, voice, uncertainty, emotion, and every factual claim.",
    "professional_register": "Rewrite in a concise professional institutional register. Preserve every fact, uncertainty, emotion, and requested action. Add nothing.",
    "plain_language": "Rewrite in clear plain language accessible to a general reader. Preserve every fact, uncertainty, emotion, and requested action. Add nothing.",
    "voice_preserving": "Improve organization and processing ease while preserving the requester's distinctive wording, emotional force, first-person voice, and nonstandard but meaningful expression. Add nothing.",
}


def prepare(args):
    data = load_frame(args.sample)
    selected = deterministic_sample(data, args.n, args.seed, strata=["post_chatgpt"])
    conditions = dict(CONDITIONS)
    if args.dialect_template_file:
        templates = pd.read_csv(args.dialect_template_file)
        required = {"template_name", "instruction"}
        if not required.issubset(templates):
            raise KeyError(f"Dialect template file must contain {sorted(required)}")
        for row in templates.itertuples():
            conditions[f"reviewed_dialect_{row.template_name}"] = str(row.instruction)
    rows = []
    for index, row in selected.iterrows():
        original = str(row["text"])
        for condition, instruction in conditions.items():
            rows.append(
                {
                    "prompt_id": f"{args.dataset}_register_{index:05d}_{condition}",
                    "research_line": 6,
                    "dataset": args.dataset,
                    "record_id": row.get("record_id", index),
                    "condition": condition,
                    "original_text": original,
                    "prompt": f"{instruction}\n\nReturn only the revised request.\n\nREQUEST:\n{original}",
                }
            )
    output = Path(args.output)
    ensure_directory(output.parent)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows):,} register-audit prompts to {output}", flush=True)


def load_generations(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def response_predictions(generations, historical, dataset):
    outcome = PRIMARY_OUTCOME.get(dataset)
    if not outcome or outcome not in historical or historical[outcome].dropna().nunique() < 2:
        return np.full(len(generations), np.nan), np.full(len(generations), np.nan)
    audit_ids = set(generations["record_id"].astype(str))
    train = historical[~historical["record_id"].astype(str).isin(audit_ids)].dropna(subset=[outcome, "text"])
    if len(train) < 500 or train[outcome].nunique() < 2:
        return np.full(len(generations), np.nan), np.full(len(generations), np.nan)
    model = make_pipeline(
        TfidfVectorizer(min_df=3, max_features=30000, ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=500, class_weight="balanced"),
    )
    model.fit(train["text"].astype(str), train[outcome].astype(int))
    original = model.predict_proba(generations["original_text"].astype(str))[:, 1]
    revised = model.predict_proba(generations["generated_text"].astype(str))[:, 1]
    return original, revised


def surface_change_against_original(original, revised):
    original_raw = surface_features(original)
    revised_raw = surface_features(revised)
    original_scores = pd.DataFrame(index=original.index)
    revised_scores = pd.DataFrame(index=revised.index)
    for column in SURFACE_COMPONENTS:
        mean = original_raw[column].mean()
        sd = original_raw[column].std(ddof=0)
        sd = sd if sd > 1e-8 else 1.0
        original_scores[column] = (original_raw[column] - mean) / sd
        revised_scores[column] = (revised_raw[column] - mean) / sd
    return revised_scores.mean(axis=1).to_numpy() - original_scores.mean(axis=1).to_numpy()


def analyze(args):
    outdir = Path(args.outdir)
    ensure_directory(outdir)
    data = load_generations(args.generations)
    if "research_line" in data:
        data = data[pd.to_numeric(data["research_line"], errors="coerce").eq(6)].reset_index(drop=True)
    if "dataset" in data:
        data = data[data["dataset"].astype(str).eq(args.dataset)].reset_index(drop=True)
    historical = load_frame(args.historical_sample)
    encoder = SentenceTransformer(args.embedding_model)
    original_embeddings = encoder.encode(data["original_text"].astype(str).tolist(), batch_size=64, normalize_embeddings=True)
    revised_embeddings = encoder.encode(data["generated_text"].astype(str).tolist(), batch_size=64, normalize_embeddings=True)
    data["semantic_similarity"] = np.sum(original_embeddings * revised_embeddings, axis=1)
    data["lexical_jaccard"] = [lexical_jaccard(a, b) for a, b in zip(data["original_text"], data["generated_text"])]
    data["unsupported_number_count"] = [
        len(extract_numbers(revised) - extract_numbers(original))
        for original, revised in zip(data["original_text"], data["generated_text"])
    ]

    original_info = text_features(data["original_text"])
    revised_info = text_features(data["generated_text"])
    for component in COMPONENTS:
        data[f"delta_{component}"] = revised_info[component].to_numpy() - original_info[component].to_numpy()
    data["surface_change"] = surface_change_against_original(data["original_text"], data["generated_text"])
    original_prediction, revised_prediction = response_predictions(data, historical, args.dataset)
    data["predicted_response_original"] = original_prediction
    data["predicted_response_revised"] = revised_prediction
    data["predicted_response_change"] = revised_prediction - original_prediction
    data.to_csv(outdir / "counterfactual_register_audit_records.csv", index=False)
    data.groupby("condition", as_index=False).agg(
        n=("prompt_id", "size"),
        semantic_similarity=("semantic_similarity", "mean"),
        lexical_jaccard=("lexical_jaccard", "mean"),
        unsupported_number_rate=("unsupported_number_count", lambda values: np.mean(values > 0)),
        surface_change=("surface_change", "mean"),
        predicted_response_change=("predicted_response_change", "mean"),
    ).to_csv(outdir / "counterfactual_register_audit_summary.csv", index=False)
    print(f"Wrote Research Line 6 results to {outdir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--sample", required=True)
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--n", type=int, default=400)
    prep.add_argument("--seed", type=int, default=2026)
    prep.add_argument("--dialect-template-file", help="Optional reviewed CSV with template_name,instruction")
    prep.set_defaults(function=prepare)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--generations", required=True)
    analysis.add_argument("--historical-sample", required=True)
    analysis.add_argument("--dataset", required=True)
    analysis.add_argument("--outdir", required=True)
    analysis.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    analysis.set_defaults(function=analyze)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
