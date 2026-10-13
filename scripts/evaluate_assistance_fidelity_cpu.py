#!/usr/bin/env python3
"""Add NLI factuality and source-voice retention checks to assistance audits."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DATASETS = ("cfpb", "nhtsa", "sandiego", "foia")
MODEL_LAYOUT = {
    "Qwen2.5-7B": ".",
    "GPT-5.6-Terra": "section6_terra_v1",
    "Qwen3-4B": "section5_local_models/qwen3_4b/full/analysis",
    "Gemma-3-4B": "section5_local_models/gemma3_4b/full/analysis",
    "Mistral-7B": "section5_local_models/mistral7b_v03/full/analysis",
    "Llama3.1-8B": "section5_local_models/llama31_8b/full/analysis",
}
TASKS = {
    "register": ("line6_register_audit", "counterfactual_register_audit_records.csv", "condition", "generated_text"),
    "assistance": ("line8_assistance", "requester_assistance_simulation_records.csv", "policy", "revision"),
}
MARKERS = {
    "first_person": re.compile(r"\b(?:i|me|my|mine|we|us|our|ours)\b", re.I),
    "contraction": re.compile(r"\b\w+(?:n't|'re|'ve|'ll|'d|'m|'s)\b", re.I),
    "uncertainty": re.compile(r"\b(?:maybe|perhaps|possibly|seem|appears?|unsure|uncertain|believe|think)\b", re.I),
    "affect": re.compile(r"\b(?:afraid|angry|anxious|concerned|confused|disappointed|frustrated|scared|upset|worried)\b", re.I),
    "emphasis": re.compile(r"(?:!|\?{2,}|\.{3,}|\b[A-Z]{3,}\b)"),
}


def stable(value: str, seed: int = 2026) -> int:
    return int(hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()[:16], 16)


def load_sample(root: Path, per_cell: int) -> pd.DataFrame:
    parts = []
    for model, relative_path in MODEL_LAYOUT.items():
        base = root / relative_path
        for dataset in DATASETS:
            for task, (line, filename, group_column, output_column) in TASKS.items():
                path = base / line / dataset / filename
                use = ["record_id", "original_text", group_column, output_column]
                if task == "assistance":
                    use.append("valid_json")
                frame = pd.read_csv(path, usecols=use, low_memory=False)
                if task == "assistance":
                    frame = frame[frame["valid_json"].astype(str).str.lower().isin(["true", "1", "yes"])]
                frame = frame.dropna(subset=["original_text", output_column]).copy()
                frame["source_text"] = frame["original_text"].astype(str)
                frame["revision_text"] = frame[output_column].astype(str)
                frame["condition"] = frame[group_column].astype(str)
                frame["task"] = task
                frame["dataset"] = dataset
                frame["model"] = model
                frame["_priority"] = frame.apply(
                    lambda row: stable(f"{task}|{dataset}|{model}|{row['condition']}|{row['record_id']}"), axis=1
                )
                frame = pd.concat(
                    [cell.nsmallest(per_cell, "_priority") for _, cell in frame.groupby("condition")],
                    ignore_index=True,
                )
                parts.append(frame[["record_id", "source_text", "revision_text", "condition", "task", "dataset", "model"]])
    data = pd.concat(parts, ignore_index=True)
    data.insert(0, "audit_id", [f"FID_{i:07d}" for i in range(1, len(data) + 1)])
    return data


def marker_erasure(source: str, revision: str) -> tuple[float, int]:
    retention = []
    for pattern in MARKERS.values():
        before = len(pattern.findall(source))
        if before:
            after = len(pattern.findall(revision))
            retention.append(min(after / before, 1.0))
    return (1.0 - float(np.mean(retention)), len(retention)) if retention else (np.nan, 0)


def add_linkability(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["source_linkability_top1"] = np.nan
    for _, index in data.groupby(["task", "dataset", "model", "condition"]).groups.items():
        cell = data.loc[index].reset_index()
        if len(cell) < 5:
            continue
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)
        matrix = vectorizer.fit_transform(pd.concat([cell["source_text"], cell["revision_text"]], ignore_index=True))
        n = len(cell)
        similarity = cosine_similarity(matrix[n:], matrix[:n])
        predicted = similarity.argmax(axis=1)
        values = predicted == np.arange(n)
        data.loc[cell["index"], "source_linkability_top1"] = values.astype(float)
    return data


def label_indices(model) -> tuple[int, int, int | None]:
    labels = {i: str(model.config.id2label.get(i, f"label_{i}")).lower() for i in range(model.config.num_labels)}
    entail = next((i for i, label in labels.items() if "entail" in label), None)
    contradiction = next((i for i, label in labels.items() if "contrad" in label), None)
    neutral = next((i for i, label in labels.items() if "neutral" in label), None)
    if entail is None or contradiction is None:
        if model.config.num_labels != 3:
            raise ValueError(f"Cannot infer NLI labels from {labels}")
        contradiction, neutral, entail = 0, 1, 2
    return entail, contradiction, neutral


def score_nli(data: pd.DataFrame, repository: str, batch_size: int, threads: int, output: Path) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(repository)
    model = AutoModelForSequenceClassification.from_pretrained(repository).eval().to("cpu")
    entail, contradiction, neutral = label_indices(model)
    torch.set_num_threads(threads)
    for column in ["source_entails_revision", "source_contradicts_revision", "revision_entails_source"]:
        if column not in data:
            data[column] = np.nan
    pending = data["source_entails_revision"].isna()
    indices = list(data.index[pending])
    for start in range(0, len(indices), batch_size):
        batch_index = indices[start : start + batch_size]
        source = data.loc[batch_index, "source_text"].tolist()
        revision = data.loc[batch_index, "revision_text"].tolist()
        forward = tokenizer(source, revision, padding=True, truncation=True, max_length=512, return_tensors="pt")
        backward = tokenizer(revision, source, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.inference_mode():
            fprob = torch.softmax(model(**forward).logits, -1).cpu().numpy()
            bprob = torch.softmax(model(**backward).logits, -1).cpu().numpy()
        data.loc[batch_index, "source_entails_revision"] = fprob[:, entail]
        data.loc[batch_index, "source_contradicts_revision"] = fprob[:, contradiction]
        data.loc[batch_index, "revision_entails_source"] = bprob[:, entail]
        if start % (batch_size * 25) == 0 or start + batch_size >= len(indices):
            data.to_csv(output, index=False, compression="gzip")
            print(f"NLI: {min(start + batch_size, len(indices)):,}/{len(indices):,}", flush=True)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root containing generated audit outputs")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--per-cell", type=int, default=100)
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-nli", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    record_path = outdir / "assistance_fidelity_records.csv.gz"
    if record_path.exists():
        data = pd.read_csv(record_path, low_memory=False)
    else:
        data = load_sample(Path(args.root), args.per_cell)
        marker = data.apply(lambda row: marker_erasure(row["source_text"], row["revision_text"]), axis=1)
        data["marker_erasure_rate"] = [value[0] for value in marker]
        data["source_marker_types"] = [value[1] for value in marker]
        data = add_linkability(data)
        data.to_csv(record_path, index=False, compression="gzip")
    if not args.skip_nli:
        data = score_nli(data, args.nli_model, args.batch_size, args.threads, record_path)
    metrics = ["marker_erasure_rate", "source_linkability_top1"]
    metrics += [column for column in ["source_entails_revision", "source_contradicts_revision", "revision_entails_source"] if column in data]
    summary = data.groupby(["task", "model", "dataset", "condition"], as_index=False)[metrics].mean()
    summary["n"] = data.groupby(["task", "model", "dataset", "condition"]).size().values
    summary.to_csv(outdir / "assistance_fidelity_summary.csv", index=False)

    link = summary[summary["task"].eq("register")].copy()
    baseline = link[link["condition"].eq("minimal_correction")][["model", "dataset", "source_linkability_top1"]].rename(columns={"source_linkability_top1": "minimal_linkability"})
    link = link.merge(baseline, on=["model", "dataset"], how="left")
    link["adapted_source_erasure"] = 1.0 - link["source_linkability_top1"] / link["minimal_linkability"].replace(0, np.nan)
    link.to_csv(outdir / "assistance_source_linkability_loss.csv", index=False)
    print(f"Wrote assistance fidelity checks to {outdir}")


if __name__ == "__main__":
    main()
