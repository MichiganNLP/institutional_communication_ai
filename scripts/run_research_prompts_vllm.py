#!/usr/bin/env python3
"""Checkpointed vLLM generation for Research Lines 6-8 prompt tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams


def completed_ids(path):
    observed = set()
    if not path.exists():
        return observed
    with path.open() as handle:
        for line in handle:
            try:
                observed.add(str(json.loads(line)["prompt_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return observed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    prompts = pd.concat([pd.read_csv(path, low_memory=False) for path in args.prompts], ignore_index=True)
    required = {"prompt_id", "prompt"}
    if not required.issubset(prompts):
        raise KeyError(f"Prompt table must contain {sorted(required)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(output)
    prompts = prompts[~prompts["prompt_id"].astype(str).isin(done)].reset_index(drop=True)
    if not len(prompts):
        print("All prompts are already complete", flush=True)
        return

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )
    with output.open("a") as handle:
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts.iloc[start : start + args.batch_size]
            messages = [
                [{"role": "system", "content": "Follow the task exactly. Never invent facts."}, {"role": "user", "content": value}]
                for value in batch["prompt"].astype(str)
            ]
            rendered = [tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True) for item in messages]
            outputs = llm.generate(rendered, sampling)
            for (_, row), generated in zip(batch.iterrows(), outputs):
                record = {}
                for key, value in row.drop(labels=["prompt"]).to_dict().items():
                    if isinstance(value, np.generic):
                        value = value.item()
                    if pd.isna(value):
                        value = None
                    record[key] = value
                record["prompt_id"] = str(record["prompt_id"])
                record["model"] = args.model
                record["generated_text"] = generated.outputs[0].text.strip()
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                handle.flush()
            print(f"completed {min(start + args.batch_size, len(prompts)):,}/{len(prompts):,}", flush=True)


if __name__ == "__main__":
    main()
