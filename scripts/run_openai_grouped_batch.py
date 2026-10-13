#!/usr/bin/env python3
"""Run prompt tables through the OpenAI Batch API in recoverable item bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI


TERMINAL = {"completed", "failed", "expired", "cancelled"}
DELIMITERS = ("\n\nREQUEST:\n", "\n\nDRAFT:\n")


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def extract_output_text(body):
    texts = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                texts.append(content["text"])
    return "\n".join(texts).strip()


def parse_json(value):
    text = str(value).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def split_prompt(prompt):
    text = str(prompt)
    for delimiter in DELIMITERS:
        if delimiter in text:
            instruction, source = text.split(delimiter, 1)
            return instruction.strip(), source.strip()
    return "Follow the task in each item independently.", text


def load_existing(path):
    completed = {}
    if not path.exists():
        return completed
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("prompt_id") and record.get("generated_text") is not None:
                    completed[str(record["prompt_id"])] = record
            except json.JSONDecodeError:
                continue
    return completed


def prompt_fingerprint(prompts):
    digest = hashlib.sha256()
    for row in prompts[["prompt_id", "prompt"]].itertuples(index=False):
        digest.update(str(row.prompt_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.prompt).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def make_group_prompt(instruction, items):
    payload = [{"id": item["prompt_id"], "text": item["source_text"]} for item in items]
    return (
        f"Apply the shared task below independently to all {len(items)} inputs. Do not transfer facts "
        "between inputs. Return every input exactly once and preserve its id.\n\n"
        "SHARED TASK:\n"
        f"{instruction}\n\n"
        "OUTPUT FORMAT:\n"
        "Return one valid JSON object only: "
        '{"items":[{"id":"input id","output":"task result"}]}. '
        "For a task requesting a JSON result, output may be that JSON object instead of a string. "
        "For a rewriting task, output must be the revised text string. Do not omit any item.\n\n"
        "INPUTS:\n"
        + json.dumps(payload, ensure_ascii=True)
    )


def build_groups(prompts, group_size, max_input_chars):
    prepared = []
    for _, row in prompts.iterrows():
        instruction, source = split_prompt(row["prompt"])
        prepared.append(
            {
                "prompt_id": str(row["prompt_id"]),
                "dataset": str(row.get("dataset", "unknown")),
                "research_line": str(row.get("research_line", "unknown")),
                "instruction": instruction,
                "source_text": source,
            }
        )

    buckets = {}
    for item in prepared:
        key = (item["dataset"], item["research_line"], item["instruction"])
        buckets.setdefault(key, []).append(item)

    groups = []
    group_number = 0
    for key in sorted(buckets):
        current = []
        current_chars = 0
        for item in buckets[key]:
            item_chars = len(item["source_text"])
            if current and (len(current) >= group_size or current_chars + item_chars > max_input_chars):
                group_number += 1
                groups.append((f"group_{group_number:05d}", key[2], current))
                current = []
                current_chars = 0
            current.append(item)
            current_chars += item_chars
        if current:
            group_number += 1
            groups.append((f"group_{group_number:05d}", key[2], current))
    return groups


def estimate_output_tokens(prompts):
    estimates = []
    for _, row in prompts.iterrows():
        _, source = split_prompt(row["prompt"])
        line_value = pd.to_numeric(row.get("research_line"), errors="coerce")
        line = int(line_value) if pd.notna(line_value) else 0
        multiplier = {6: 1.05, 7: 1.35, 8: 1.40}.get(line, 1.25)
        overhead = {6: 20, 7: 100, 8: 120}.get(line, 80)
        estimates.append(math.ceil(len(source) / 4 * multiplier + overhead))
    return sum(estimates)


def print_preflight(prompts, missing, groups, args, workdir):
    input_tokens = sum(math.ceil(len(make_group_prompt(instruction, items)) / 4) for _, instruction, items in groups)
    output_tokens = estimate_output_tokens(missing)
    empirical_cost = None
    if args.observed_cost_usd is not None and args.observed_items:
        average_cost = args.observed_cost_usd / args.observed_items
        materialized = prompts[~prompts["prompt_id"].isin(missing["prompt_id"])].reset_index(drop=True)
        length_factor = 1.0
        if not materialized.empty and not missing.empty:
            missing_units = sum(math.ceil(len(value) / 4) for value in missing["prompt"].astype(str)) + output_tokens
            materialized_units = (
                sum(math.ceil(len(value) / 4) for value in materialized["prompt"].astype(str))
                + estimate_output_tokens(materialized)
            )
            length_factor = (missing_units / len(missing)) / (materialized_units / len(materialized))
        empirical_cost = len(missing) * average_cost * length_factor
    rate_cost = None
    if args.input_usd_per_million is not None and args.output_usd_per_million is not None:
        rate_cost = (
            input_tokens * args.input_usd_per_million
            + output_tokens * args.output_usd_per_million
        ) / 1_000_000
    estimate = empirical_cost if empirical_cost is not None else rate_cost
    estimate_with_contingency = estimate * 1.25 if estimate is not None else None
    report = {
        "total_items": len(prompts),
        "already_materialized": len(prompts) - len(missing),
        "missing_items": len(missing),
        "api_requests": len(groups),
        "target_group_size": args.group_size,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "empirical_cost_usd": empirical_cost,
        "rate_based_cost_usd": rate_cost,
        "enforced_cost_estimate_usd": estimate,
        "cost_estimate_with_25pct_contingency_usd": estimate_with_contingency,
        "max_estimated_cost_usd": args.max_estimated_cost_usd,
    }
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if estimate is None and args.submit:
        raise RuntimeError("No dollar estimate is available; provide observed cost/items or token rates")
    if estimate_with_contingency is not None and estimate_with_contingency > args.max_estimated_cost_usd and args.submit:
        raise RuntimeError(
            f"Estimated cost with contingency ${estimate_with_contingency:.2f} exceeds cap "
            f"${args.max_estimated_cost_usd:.2f}; not submitting"
        )


def download(client, file_id, path):
    response = client.files.content(file_id)
    if hasattr(response, "write_to_file"):
        response.write_to_file(path)
    else:
        path.write_bytes(response.content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--group-size", type=int, default=25)
    parser.add_argument("--max-input-chars", type=int, default=80000)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--observed-cost-usd", type=float)
    parser.add_argument("--observed-items", type=int)
    parser.add_argument("--input-usd-per-million", type=float)
    parser.add_argument("--output-usd-per-million", type=float)
    parser.add_argument("--max-estimated-cost-usd", type=float, default=12.0)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.group_size <= 30:
        raise ValueError("--group-size must be between 1 and 30")
    workdir = Path(args.workdir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompts = pd.concat([pd.read_csv(path, low_memory=False) for path in args.prompts], ignore_index=True)
    prompts["prompt_id"] = prompts["prompt_id"].astype(str)
    if prompts["prompt_id"].duplicated().any():
        raise ValueError("prompt_id must be unique across all prompt files")
    metadata = {str(row["prompt_id"]): row for _, row in prompts.iterrows()}
    completed = load_existing(output_path)
    valid_ids = set(metadata)
    completed = {key: value for key, value in completed.items() if key in valid_ids}
    missing = prompts[~prompts["prompt_id"].isin(completed)].reset_index(drop=True)
    groups = build_groups(missing, args.group_size, args.max_input_chars)
    print_preflight(prompts, missing, groups, args, workdir)

    if missing.empty:
        print(f"All {len(prompts):,} items are already materialized; no API request needed", flush=True)
        return
    if not args.submit:
        print("Dry run only. Add --submit to create the remote batch.", flush=True)
        return
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    state_path = workdir / "batch_state.json"
    request_path = workdir / "batch_requests.jsonl"
    raw_output_path = workdir / "batch_output.jsonl"
    raw_error_path = workdir / "batch_errors.jsonl"
    fingerprint = prompt_fingerprint(missing)
    group_members = {group_id: [item["prompt_id"] for item in items] for group_id, _, items in groups}

    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError("Existing grouped batch state does not match the current missing-item set")
        batch_id = state["batch_id"]
        print(f"Resuming grouped OpenAI batch {batch_id}", flush=True)
    else:
        workdir.mkdir(parents=True, exist_ok=True)
        with request_path.open("w") as handle:
            for group_id, instruction, items in groups:
                request = {
                    "custom_id": group_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": args.model,
                        "instructions": (
                            "Process each supplied item independently. Never transfer, add, or infer facts. "
                            "Return valid JSON in the exact requested wrapper."
                        ),
                        "input": make_group_prompt(instruction, items),
                        "reasoning": {"effort": args.reasoning_effort},
                        "max_output_tokens": args.max_output_tokens,
                        "store": False,
                    },
                }
                handle.write(json.dumps(request, ensure_ascii=True) + "\n")
        with request_path.open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"task": "section6_grouped_recovery", "model": args.model},
        )
        batch_id = batch.id
        state_path.write_text(
            json.dumps(
                {
                    "batch_id": batch.id,
                    "input_file_id": uploaded.id,
                    "model": args.model,
                    "n_items": len(missing),
                    "n_requests": len(groups),
                    "fingerprint": fingerprint,
                    "group_members": group_members,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Submitted {len(groups):,} grouped requests for {len(missing):,} items: {batch_id}", flush=True)

    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"status={batch.status} completed={getattr(counts, 'completed', 0)} "
            f"failed={getattr(counts, 'failed', 0)} total={getattr(counts, 'total', len(groups))}",
            flush=True,
        )
        if batch.status in TERMINAL:
            break
        time.sleep(args.poll_seconds)

    if batch.output_file_id:
        download(client, batch.output_file_id, raw_output_path)
    if batch.error_file_id:
        download(client, batch.error_file_id, raw_error_path)
    if not raw_output_path.exists():
        raise RuntimeError(f"Grouped batch ended with status={batch.status} and no output file")

    invalid_groups = []
    unexpected_ids = []
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "responses_with_usage": 0}
    with raw_output_path.open() as handle:
        for line in handle:
            item = json.loads(line)
            group_id = str(item["custom_id"])
            response = item.get("response") or {}
            body = response.get("body") or {}
            usage = body.get("usage") or {}
            if usage:
                usage_totals["responses_with_usage"] += 1
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    usage_totals[key] += int(usage.get(key, 0) or 0)
            parsed = parse_json(extract_output_text(body))
            results = parsed.get("items") if isinstance(parsed, dict) else None
            if response.get("status_code") != 200 or not isinstance(results, list):
                invalid_groups.append(group_id)
                continue
            allowed = set(group_members.get(group_id, []))
            for result in results:
                if not isinstance(result, dict):
                    continue
                prompt_id = str(result.get("id", result.get("prompt_id", "")))
                if prompt_id not in allowed or prompt_id not in metadata:
                    unexpected_ids.append(prompt_id)
                    continue
                generated = result.get("output")
                if generated is None:
                    continue
                generated_text = generated if isinstance(generated, str) else json.dumps(generated, ensure_ascii=True)
                row = metadata[prompt_id]
                record = {key: json_value(value) for key, value in row.drop(labels=["prompt"]).to_dict().items()}
                record.update(
                    {
                        "prompt_id": prompt_id,
                        "model": args.model,
                        "generated_text": generated_text,
                        "openai_response_id": body.get("id"),
                        "openai_request_id": response.get("request_id"),
                        "batch_group_id": group_id,
                    }
                )
                completed[prompt_id] = record

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for prompt_id in sorted(completed):
            handle.write(json.dumps(completed[prompt_id], ensure_ascii=True) + "\n")
    temporary.replace(output_path)
    remaining = sorted(valid_ids - set(completed))
    report = {
        "materialized": len(completed),
        "expected": len(prompts),
        "remaining": len(remaining),
        "invalid_groups": invalid_groups,
        "unexpected_ids": unexpected_ids,
        "remaining_ids": remaining,
        "usage": usage_totals,
    }
    (workdir / "materialization_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Materialized {len(completed):,}/{len(prompts):,} item outputs at {output_path}", flush=True)
    if remaining and not args.allow_partial:
        raise RuntimeError(f"{len(remaining):,} items remain; rerun them from a fresh workdir")


if __name__ == "__main__":
    main()
