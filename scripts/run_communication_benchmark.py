#!/usr/bin/env python3
"""Research Line 7: build and evaluate a requester-side communication benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_extension_common import (
    MOVE_NAMES,
    MOVE_PATTERNS,
    deterministic_sample,
    ensure_directory,
    extract_numbers,
    lexical_jaccard,
    load_frame,
    parse_json_object,
)


def remove_move(text, move):
    match = MOVE_PATTERNS[move].search(str(text))
    if not match:
        return None, None
    start = max(0, str(text).rfind(".", 0, match.start()) + 1)
    endings = [position for position in [str(text).find(mark, match.end()) for mark in ".!?\n"] if position >= 0]
    end = min(endings) + 1 if endings else match.end()
    deleted = str(text)[start:end].strip()
    corrupted = (str(text)[:start] + " " + str(text)[end:]).strip()
    return corrupted, deleted


def prepare(args):
    data = deterministic_sample(load_frame(args.sample), args.pool_size, args.seed, strata=["post_chatgpt"])
    rows = []
    for index, row in data.iterrows():
        available = [move for move in MOVE_NAMES if MOVE_PATTERNS[move].search(str(row["text"]))]
        if not available:
            continue
        move = available[index % len(available)]
        corrupted, deleted = remove_move(row["text"], move)
        if not corrupted or not deleted or len(deleted.split()) < 2:
            continue
        prompt = (
            "A requester draft is missing important institutional information. Diagnose the missing information "
            "without inventing it. Return valid JSON with keys missing_moves (a list using only these labels: "
            f"{', '.join(MOVE_NAMES)}), clarifying_question, and revision. In revision, preserve the draft and use "
            "a bracketed placeholder for unavailable facts.\n\nDRAFT:\n" + corrupted
        )
        rows.append(
            {
                "prompt_id": f"{args.dataset}_benchmark_{len(rows):05d}",
                "research_line": 7,
                "dataset": args.dataset,
                "record_id": row.get("record_id", index),
                "post_chatgpt": row.get("post_chatgpt"),
                "missing_move": move,
                "deleted_span": deleted,
                "original_text": row["text"],
                "corrupted_text": corrupted,
                "prompt": prompt,
            }
        )
        if len(rows) >= args.n:
            break
    output = Path(args.output)
    ensure_directory(output.parent)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows):,} benchmark prompts to {output}", flush=True)


def load_generations(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def analyze(args):
    outdir = Path(args.outdir)
    ensure_directory(outdir)
    data = load_generations(args.generations)
    if "research_line" in data:
        data = data[pd.to_numeric(data["research_line"], errors="coerce").eq(7)].reset_index(drop=True)
    if args.dataset and "dataset" in data:
        data = data[data["dataset"].astype(str).eq(args.dataset)].reset_index(drop=True)
    parsed = [parse_json_object(value) for value in data["generated_text"]]
    data["valid_json"] = [item is not None for item in parsed]
    data["predicted_moves"] = [
        json.dumps(item.get("missing_moves", [])) if isinstance(item, dict) else "[]" for item in parsed
    ]
    data["diagnosis_correct"] = [
        bool(item and row.missing_move in [str(value) for value in item.get("missing_moves", [])])
        for item, row in zip(parsed, data.itertuples())
    ]
    revisions = [item.get("revision", "") if isinstance(item, dict) else "" for item in parsed]
    questions = [item.get("clarifying_question", "") if isinstance(item, dict) else "" for item in parsed]
    data["revision"] = revisions
    data["clarifying_question"] = questions
    data["question_mentions_missing_move"] = [
        row.missing_move.replace("_", " ") in str(question).lower().replace("_", " ")
        for row, question in zip(data.itertuples(), questions)
    ]
    data["unsupported_number_count"] = [
        len(extract_numbers(revision) - extract_numbers(corrupted))
        for revision, corrupted in zip(revisions, data["corrupted_text"])
    ]
    data["revision_lexical_jaccard"] = [
        lexical_jaccard(corrupted, revision) for corrupted, revision in zip(data["corrupted_text"], revisions)
    ]
    data.to_csv(outdir / "communication_benchmark_records.csv", index=False)
    pd.DataFrame(
        [
            {
                "n": len(data),
                "valid_json_rate": data["valid_json"].mean(),
                "missing_move_accuracy": data["diagnosis_correct"].mean(),
                "targeted_question_rate": data["question_mentions_missing_move"].mean(),
                "unsupported_number_rate": (data["unsupported_number_count"] > 0).mean(),
                "mean_revision_lexical_jaccard": data["revision_lexical_jaccard"].mean(),
            }
        ]
    ).to_csv(outdir / "communication_benchmark_summary.csv", index=False)
    data.groupby("missing_move", as_index=False).agg(
        n=("prompt_id", "size"),
        accuracy=("diagnosis_correct", "mean"),
        targeted_question_rate=("question_mentions_missing_move", "mean"),
        unsupported_number_rate=("unsupported_number_count", lambda values: np.mean(values > 0)),
    ).to_csv(outdir / "communication_benchmark_by_move.csv", index=False)
    print(f"Wrote Research Line 7 results to {outdir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--sample", required=True)
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--n", type=int, default=500)
    prep.add_argument("--pool-size", type=int, default=5000)
    prep.add_argument("--seed", type=int, default=2026)
    prep.set_defaults(function=prepare)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--generations", required=True)
    analysis.add_argument("--dataset")
    analysis.add_argument("--outdir", required=True)
    analysis.set_defaults(function=analyze)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
