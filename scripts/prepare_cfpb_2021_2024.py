#!/usr/bin/env python3
"""
Download the CFPB bulk complaint CSV and create a 2021-2024 public-narrative CSV.

This creates the file expected by build_analytic_sample_attrition.py:

    data/raw/cfpb_full_2021_2024.csv

The CFPB bulk file is large. The script streams the zipped CSV in chunks rather
than loading the full 17M+ row file into memory.
"""

import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


CFPB_CSV_ZIP_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
DEFAULT_KEEP_COLUMNS = [
    "Complaint ID",
    "Date received",
    "Product",
    "Issue",
    "Sub-product",
    "Sub-issue",
    "Company",
    "Company response to consumer",
    "Timely response?",
    "Submitted via",
    "State",
    "ZIP code",
    "Tags",
    "Consumer complaint narrative",
]


def download_file(url, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("downloading {} -> {}".format(url, out_path))
    with urllib.request.urlopen(url, timeout=120) as resp, out_path.open("wb") as f:
        shutil.copyfileobj(resp, f, length=1024 * 1024)
    print("downloaded {}".format(out_path))


def get_csv_member(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        csv_members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
    if len(csv_members) != 1:
        raise RuntimeError("expected one CSV in {}, found {}".format(zip_path, csv_members))
    return csv_members[0]


def filter_bulk_csv(zip_path, out_path, start_date, end_date, chunksize):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    csv_member = get_csv_member(zip_path)
    total_rows = 0
    kept_rows = 0
    first_write = True

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(csv_member) as f:
            reader = pd.read_csv(f, dtype=str, chunksize=chunksize, usecols=lambda c: c in DEFAULT_KEEP_COLUMNS)
            for chunk_id, chunk in enumerate(reader, start=1):
                total_rows += len(chunk)
                dates = pd.to_datetime(chunk["Date received"], errors="coerce")
                narrative = chunk["Consumer complaint narrative"].fillna("").astype(str).str.strip()
                mask = (
                    (dates >= pd.Timestamp(start_date))
                    & (dates <= pd.Timestamp(end_date))
                    & narrative.ne("")
                    & narrative.str.lower().ne("nan")
                )
                filtered = chunk.loc[mask].copy()
                kept_rows += len(filtered)

                filtered.to_csv(out_path, mode="w" if first_write else "a", header=first_write, index=False)
                first_write = False

                print(
                    "chunk {}: read {:,}, kept {:,}, cumulative kept {:,}".format(
                        chunk_id, len(chunk), len(filtered), kept_rows
                    )
                )

    print("read {:,} rows".format(total_rows))
    print("kept {:,} narrative rows from {} to {}".format(kept_rows, start_date, end_date))
    print("wrote {}".format(out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-dir", default=None, help="Directory for large CFPB files.")
    parser.add_argument("--zip", default=None, help="Local CFPB bulk zip path.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--chunksize", type=int, default=250000)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true", help="Use an existing --zip file only.")
    args = parser.parse_args()

    scratch_dir = args.scratch_dir or os.environ.get("CFPB_SCRATCH_DIR")
    if scratch_dir:
        base_dir = Path(scratch_dir)
        default_zip = base_dir / "complaints.csv.zip"
        default_out = base_dir / "cfpb_full_2021_2024.csv"
    else:
        default_zip = Path("data/raw/complaints.csv.zip")
        default_out = Path("data/raw/cfpb_full_2021_2024.csv")

    zip_path = Path(args.zip) if args.zip else default_zip
    out_path = Path(args.out) if args.out else default_out

    if not args.skip_download and (args.force_download or not zip_path.exists()):
        download_file(CFPB_CSV_ZIP_URL, zip_path)

    if not zip_path.exists():
        raise FileNotFoundError("missing CFPB zip: {}".format(zip_path))

    filter_bulk_csv(zip_path, out_path, args.start_date, args.end_date, args.chunksize)


if __name__ == "__main__":
    main()
