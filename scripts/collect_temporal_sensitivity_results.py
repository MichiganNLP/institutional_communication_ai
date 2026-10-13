#!/usr/bin/env python3
"""Verify and collect the already-computed temporal sensitivity suite."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


DATASETS = ("cfpb", "nhtsa", "sandiego", "foia")
REQUIRED = (
    "temporal_actual_2022_11_actionability_exclude0.csv",
    "temporal_actual_2022_11_actionability_exclude1.csv",
    "temporal_actual_2022_11_actionability_exclude3.csv",
    "temporal_actual_2022_11_actionability_balanced.csv",
    "temporal_placebo_2021_11_actionability_exclude0.csv",
    "temporal_placebo_2022_05_actionability_exclude0.csv",
    "temporal_alternative_2023_01_actionability_exclude0.csv",
    "temporal_alternative_2023_03_actionability_exclude0.csv",
    "temporal_model_index.csv",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrected-root", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    corrected = Path(args.corrected_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    inventory = []
    combined = []

    for dataset in DATASETS:
        source = corrected / dataset / "models"
        destination = outdir / dataset
        destination.mkdir(parents=True, exist_ok=True)
        missing = [name for name in REQUIRED if not (source / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{dataset} is missing temporal outputs: {missing}")

        for path in sorted(source.glob("temporal_*.csv")):
            target = destination / path.name
            shutil.copy2(path, target)
            inventory.append(
                {
                    "dataset": dataset,
                    "file": path.name,
                    "source": str(path),
                    "collected_copy": str(target),
                    "bytes": path.stat().st_size,
                }
            )
        index = pd.read_csv(source / "temporal_model_index.csv", low_memory=False)
        if "dataset" not in index:
            index.insert(0, "dataset", dataset)
        combined.append(index)

    pd.DataFrame(inventory).to_csv(outdir / "temporal_sensitivity_inventory.csv", index=False)
    pd.concat(combined, ignore_index=True).to_csv(
        outdir / "temporal_sensitivity_combined.csv", index=False
    )
    print(pd.DataFrame(inventory).groupby("dataset").size().to_string())
    print(f"Collected {len(inventory):,} verified temporal outputs in {outdir}")


if __name__ == "__main__":
    main()
