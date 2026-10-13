#!/usr/bin/env python3
"""Download narrative-heavy complaint/request datasets.

Supported datasets:
- sandiego311: San Diego Get It Done yearly closed-service CSVs.
- nhtsa: NHTSA ODI five-year complaint flat file, filtered to requested years.
- foia: public FOIA log pages; downloads CSV/XLSX links whose text/URL mention
  requested years and combines parseable files.

FOIA agencies vary a lot in format. This script deliberately keeps raw files and
adds source metadata so failed agency-specific files can be inspected without
blocking the other sources.
"""

import argparse
import csv
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd


UA = "Mozilla/5.0 ai-agents-inequality-research"

NHTSA_2020_2024 = "https://static.nhtsa.gov/odi/ffdd/cmpl/COMPLAINTS_RECEIVED_2020-2024.zip"
NHTSA_2025_2026 = "https://static.nhtsa.gov/odi/ffdd/cmpl/COMPLAINTS_RECEIVED_2025-2026.zip"

NHTSA_COLUMNS = [
    "cmplid",
    "odi_number",
    "manufacturer",
    "make",
    "model",
    "model_year",
    "crash",
    "date_of_incident",
    "fire",
    "number_of_injuries",
    "number_of_deaths",
    "component",
    "city",
    "state",
    "vin",
    "date_complaint_filed",
    "date_added",
    "mileage",
    "failure_speed",
    "summary",
]

FOIA_SOURCES = {
    "irs": "https://www.irs.gov/privacy-disclosure/irs-foia-logs",
    "doj_civil": "https://www.justice.gov/civil/foia-library/logs",
    "doj_oip": "https://www.justice.gov/oip/available-documents-oip",
    "education": "https://www.ed.gov/about/ed-overview/required-notices/freedom-of-information-act-foia/foia-logs",
    "gsa": "https://www.gsa.gov/reference/freedom-of-information-act-foia/reports",
}


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self._href = None
        self._text = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            attrs = dict(attrs)
            self._href = attrs.get("href")
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._href is not None:
            self.links.append((self._href, " ".join(self._text).strip()))
            self._href = None
            self._text = []


def download_url(url, outpath):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp, outpath.open("wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as exc:
        print(f"urllib download failed ({exc}); retrying with curl", flush=True)
        cmd = [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "5",
            "--retry-delay",
            "10",
            "--user-agent",
            UA,
            "--output",
            str(outpath),
            url,
        ]
        subprocess.run(cmd, check=True)


def fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode("utf-8", errors="replace")


def concat_csvs(paths, combined):
    expected = None
    total = 0
    combined.parent.mkdir(parents=True, exist_ok=True)
    with combined.open("w", newline="", encoding="utf-8") as out:
        writer = None
        for path in paths:
            with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    continue
                header = [h.strip().lower() for h in header]
                if expected is None:
                    expected = header
                    writer = csv.writer(out)
                    writer.writerow(header)
                elif header != expected:
                    raise RuntimeError(f"Header mismatch in {path}")
                n = 0
                for row in reader:
                    writer.writerow(row)
                    n += 1
                total += n
                print(f"Added {n:,} rows from {path.name}", flush=True)
    print(f"Wrote {total:,} rows to {combined}", flush=True)


def download_sandiego(years, outdir, combined):
    paths = []
    for year in years:
        path = outdir / f"sandiego311_{year}.csv"
        paths.append(path)
        if not path.exists() or path.stat().st_size == 0:
            url = f"https://seshat.datasd.org/get_it_done_reports/get_it_done_requests_closed_{year}_datasd.csv"
            print(f"Downloading San Diego {year}: {url}", flush=True)
            download_url(url, path)
        else:
            print(f"Using existing {path}", flush=True)
    concat_csvs(paths, combined)


def parse_nhtsa_date(s):
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return pd.NaT
    return pd.to_datetime(s, errors="coerce")


def download_nhtsa(years, outdir, combined):
    urls = [NHTSA_2020_2024]
    if any(y >= 2025 for y in years):
        urls.append(NHTSA_2025_2026)
    parts = []
    for url in urls:
        zip_path = outdir / Path(urllib.parse.urlparse(url).path).name
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            print(f"Downloading NHTSA: {url}", flush=True)
            download_url(url, zip_path)
        else:
            print(f"Using existing {zip_path}", flush=True)

        with zipfile.ZipFile(zip_path) as zf:
            txt_members = [m for m in zf.namelist() if m.lower().endswith(".txt")]
            if not txt_members:
                raise RuntimeError(f"No txt file in {zip_path}")
            member = txt_members[0]
            print(f"Reading {member}", flush=True)
            with zf.open(member) as f:
                # NHTSA flat files are tab-delimited and have no header.
                df = pd.read_csv(
                    f,
                    sep="\t",
                    header=None,
                    dtype=str,
                    encoding_errors="replace",
                    engine="python",
                    quoting=csv.QUOTE_NONE,
                )
                if df.shape[1] < len(NHTSA_COLUMNS):
                    raise RuntimeError(f"NHTSA file has only {df.shape[1]} columns")
                df = df.iloc[:, : len(NHTSA_COLUMNS)]
                df.columns = NHTSA_COLUMNS
        filed = df["date_complaint_filed"].apply(parse_nhtsa_date)
        df = df[filed.dt.year.isin(years)].copy()
        df["date_complaint_filed_parsed"] = filed[filed.dt.year.isin(years)].dt.strftime("%Y-%m-%d")
        part = outdir / (zip_path.stem + "_filtered.csv")
        df.to_csv(part, index=False)
        print(f"Kept {len(df):,} NHTSA rows from {zip_path.name}", flush=True)
        parts.append(part)
    concat_same_schema(parts, combined)


def concat_same_schema(paths, combined):
    combined.parent.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(p, dtype=str) for p in paths if p.exists() and p.stat().st_size > 0]
    if not frames:
        raise RuntimeError("No rows to combine")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(combined, index=False)
    print(f"Wrote {len(out):,} rows to {combined}", flush=True)


def normalize_cols(cols):
    seen = {}
    out = []
    for col in cols:
        base = re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")
        if not base:
            base = "unnamed"
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.append(base if n == 0 else f"{base}_{n + 1}")
    return out


def read_foia_file(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, encoding_errors="replace")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)
    raise ValueError(f"Unsupported FOIA file: {path}")


def link_matches_year(href, text, years):
    blob = f"{href} {text}".lower()
    return any(str(y) in blob or f"fy{str(y)[-2:]}" in blob for y in years)


def download_foia(years, outdir, combined, agencies):
    rows = []
    manifest = []
    for agency in agencies:
        index = FOIA_SOURCES[agency]
        print(f"Scraping {agency}: {index}", flush=True)
        html = fetch_text(index)
        parser = LinkParser()
        parser.feed(html)
        links = []
        for href, text in parser.links:
            if not href:
                continue
            url = urllib.parse.urljoin(index, href)
            low = url.lower().split("?")[0]
            if not low.endswith((".csv", ".xlsx", ".xls")):
                continue
            if not link_matches_year(url, text, years):
                continue
            links.append((url, text))
        print(f"{agency}: found {len(links)} parseable CSV/XLSX links", flush=True)

        for i, (url, text) in enumerate(links, start=1):
            ext = Path(urllib.parse.urlparse(url).path).suffix or ".dat"
            raw = outdir / agency / f"{agency}_{i:04d}{ext}"
            raw.parent.mkdir(parents=True, exist_ok=True)
            if not raw.exists() or raw.stat().st_size == 0:
                try:
                    download_url(url, raw)
                except Exception as exc:
                    manifest.append({"agency": agency, "url": url, "text": text, "status": f"download_failed: {exc}"})
                    continue
            try:
                df = read_foia_file(raw)
            except Exception as exc:
                manifest.append({"agency": agency, "url": url, "text": text, "path": str(raw), "status": f"parse_failed: {exc}"})
                continue
            df.columns = normalize_cols(df.columns)
            df["foia_source_agency"] = agency
            df["foia_source_url"] = url
            df["foia_source_link_text"] = text
            df["foia_source_file"] = str(raw)
            rows.append(df)
            manifest.append({"agency": agency, "url": url, "text": text, "path": str(raw), "status": "ok", "rows": len(df)})
            print(f"{agency}: parsed {len(df):,} rows from {raw.name}", flush=True)

    with (outdir / "foia_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    if not rows:
        raise RuntimeError(f"No FOIA files parsed. Inspect {outdir / 'foia_manifest.json'}")
    combined_df = pd.concat(rows, ignore_index=True, sort=False)
    combined_df.to_csv(combined, index=False)
    print(f"Wrote {len(combined_df):,} FOIA rows to {combined}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["sandiego311", "nhtsa", "foia"])
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--combined", required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    parser.add_argument("--foia-agencies", nargs="+", choices=sorted(FOIA_SOURCES), default=["irs", "doj_civil", "doj_oip", "education", "gsa"])
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    combined = Path(args.combined)
    combined.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset == "sandiego311":
        download_sandiego(args.years, outdir, combined)
    elif args.dataset == "nhtsa":
        download_nhtsa(args.years, outdir, combined)
    elif args.dataset == "foia":
        download_foia(args.years, outdir, combined, args.foia_agencies)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
