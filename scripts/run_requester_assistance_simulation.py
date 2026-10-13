#!/usr/bin/env python3
"""Research Line 8: compare requester-side AI assistance policies."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from corrected_actionability_common import COMPONENTS, text_features
from research_extension_common import (
    deterministic_sample,
    ensure_directory,
    extract_numbers,
    lexical_jaccard,
    load_frame,
    parse_json_object,
)
from run_information_polish_analysis import SURFACE_COMPONENTS, surface_features


POLICIES = {
    "polish_only": (
        "Improve grammar and organization only. Do not add facts, evidence, procedural steps, certainty, or a remedy. "
        "Return JSON with keys questions (an empty list) and revision."
    ),
    "requirements_only": (
        "Do not rewrite the request. Identify the smallest set of missing facts the relevant institution would need. "
        "Return JSON with keys questions (a list of targeted questions) and revision (the original request unchanged)."
    ),
    "targeted_elicitation": (
        "Ask targeted questions for missing dates, actors, evidence, prior contact, and requested remedy. Then provide a minimally organized revision "
        "using bracketed placeholders where answers are unavailable. Return JSON with keys questions and revision."
    ),
    "full_constrained_support": (
        "Diagnose missing information, ask only necessary questions, and revise for clarity while preserving voice, uncertainty, and emotion. "
        "Use bracketed placeholders for missing facts and never invent details. Return JSON with keys questions and revision."
    ),
}

FIRST_PERSON = re.compile(r"\b(?:I|me|my|mine|we|our|ours)\b", re.I)
AFFECT = re.compile(r"\b(?:afraid|angry|anxious|concerned|distress|frustrat|hardship|stress|urgent|worried)\w*\b", re.I)


def prepare(args):
    data = deterministic_sample(load_frame(args.sample), args.n, args.seed, strata=["post_chatgpt"])
    rows = []
    for index, row in data.iterrows():
        original = str(row["text"])
        for policy, instruction in POLICIES.items():
            rows.append(
                {
                    "prompt_id": f"{args.dataset}_assistance_{index:05d}_{policy}",
                    "research_line": 8,
                    "dataset": args.dataset,
                    "record_id": row.get("record_id", index),
                    "policy": policy,
                    "original_text": original,
                    "prompt": f"{instruction}\nReturn valid JSON only.\n\nREQUEST:\n{original}",
                }
            )
    output = Path(args.output)
    ensure_directory(output.parent)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows):,} assistance-policy prompts to {output}", flush=True)


def load_generations(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def score_surface_against_original(original, revised):
    original_raw = surface_features(original)
    revised_raw = surface_features(revised)
    original_z = pd.DataFrame(index=original.index)
    revised_z = pd.DataFrame(index=revised.index)
    for column in SURFACE_COMPONENTS:
        mean = original_raw[column].mean()
        sd = original_raw[column].std(ddof=0)
        sd = sd if sd > 1e-8 else 1.0
        original_z[column] = (original_raw[column] - mean) / sd
        revised_z[column] = (revised_raw[column] - mean) / sd
    return revised_z.mean(axis=1).to_numpy() - original_z.mean(axis=1).to_numpy()


def analyze(args):
    outdir = Path(args.outdir)
    ensure_directory(outdir)
    data = load_generations(args.generations)
    if "research_line" in data:
        data = data[pd.to_numeric(data["research_line"], errors="coerce").eq(8)].reset_index(drop=True)
    if args.dataset and "dataset" in data:
        data = data[data["dataset"].astype(str).eq(args.dataset)].reset_index(drop=True)
    parsed = [parse_json_object(value) for value in data["generated_text"]]
    data["valid_json"] = [item is not None for item in parsed]
    revisions = [item.get("revision", "") if isinstance(item, dict) else "" for item in parsed]
    if args.allow_plain_revisions:
        for index, (item, generated) in enumerate(zip(parsed, data["generated_text"])):
            if item is not None:
                continue
            candidate = str(generated).strip()
            if len(candidate) >= 20 and not candidate.startswith(("{", "[", "```")):
                revisions[index] = candidate
    data["usable_revision"] = [bool(str(value).strip()) for value in revisions]
    data["plaintext_revision_used"] = data["usable_revision"] & ~data["valid_json"]
    questions = [item.get("questions", []) if isinstance(item, dict) else [] for item in parsed]
    data["revision"] = revisions
    data["question_count"] = [len(value) if isinstance(value, list) else 0 for value in questions]
    data.loc[data["plaintext_revision_used"], "question_count"] = np.nan

    encoder = SentenceTransformer(args.embedding_model)
    original_embeddings = encoder.encode(data["original_text"].astype(str).tolist(), batch_size=64, normalize_embeddings=True)
    revised_embeddings = encoder.encode(pd.Series(revisions).astype(str).tolist(), batch_size=64, normalize_embeddings=True)
    data["semantic_similarity"] = np.sum(original_embeddings * revised_embeddings, axis=1)
    data["lexical_jaccard"] = [lexical_jaccard(a, b) for a, b in zip(data["original_text"], revisions)]
    data["unsupported_number_count"] = [
        len(extract_numbers(revision) - extract_numbers(original))
        for original, revision in zip(data["original_text"], revisions)
    ]
    original_info = text_features(data["original_text"])
    revised_info = text_features(pd.Series(revisions, index=data.index))
    for component in COMPONENTS:
        data[f"delta_{component}"] = revised_info[component].to_numpy() - original_info[component].to_numpy()
    data["diagnostic_information_change"] = data[[f"delta_{name}" for name in COMPONENTS]].mean(axis=1)
    data["surface_change"] = score_surface_against_original(data["original_text"], pd.Series(revisions, index=data.index))

    original_words = data["original_text"].str.count(r"\b\w+\b").clip(lower=1)
    revised_words = pd.Series(revisions).str.count(r"\b\w+\b").clip(lower=1)
    data["first_person_retention"] = (
        pd.Series(revisions).str.count(FIRST_PERSON) / revised_words
    ) / (data["original_text"].str.count(FIRST_PERSON) / original_words).replace(0, np.nan)
    data["affect_retention"] = (
        pd.Series(revisions).str.count(AFFECT) / revised_words
    ) / (data["original_text"].str.count(AFFECT) / original_words).replace(0, np.nan)
    rewrite_metrics = [
        "semantic_similarity", "lexical_jaccard", "unsupported_number_count",
        "diagnostic_information_change", "surface_change",
        "first_person_retention", "affect_retention",
    ]
    data.loc[~data["usable_revision"], rewrite_metrics] = np.nan
    data.to_csv(outdir / "requester_assistance_simulation_records.csv", index=False)
    data.groupby("policy", as_index=False).agg(
        n=("prompt_id", "size"),
        valid_json_rate=("valid_json", "mean"),
        usable_revision_rate=("usable_revision", "mean"),
        plaintext_revision_rate=("plaintext_revision_used", "mean"),
        mean_questions=("question_count", "mean"),
        semantic_similarity=("semantic_similarity", "mean"),
        lexical_jaccard=("lexical_jaccard", "mean"),
        unsupported_number_rate=("unsupported_number_count", lambda values: values.dropna().gt(0).mean()),
        diagnostic_information_change=("diagnostic_information_change", "mean"),
        surface_change=("surface_change", "mean"),
        first_person_retention=("first_person_retention", "median"),
        affect_retention=("affect_retention", "median"),
    ).to_csv(outdir / "requester_assistance_simulation_summary.csv", index=False)
    print(f"Wrote Research Line 8 results to {outdir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--sample", required=True)
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--n", type=int, default=400)
    prep.add_argument("--seed", type=int, default=2026)
    prep.set_defaults(function=prepare)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--generations", required=True)
    analysis.add_argument("--dataset")
    analysis.add_argument("--outdir", required=True)
    analysis.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    analysis.add_argument("--allow-plain-revisions", action="store_true")
    analysis.set_defaults(function=analyze)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
