#!/usr/bin/env python3
"""Download ACS 5-year ZCTA context variables for CFPB neighborhood analysis.

Default is 2023 ACS 5-year because it is stable and available through the Census
API. The output is one row per ZCTA with derived rates used for moderation:
income, poverty, unemployment, education, race/ethnicity composition,
foreign-born share, limited-English household share, broadband, tenure, and age.
"""

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


VARS = {
    "NAME": "name",
    "B19013_001E": "median_household_income",
    "B17001_001E": "poverty_universe",
    "B17001_002E": "poverty_count",
    "B23025_003E": "civilian_labor_force",
    "B23025_005E": "unemployed_count",
    "B15003_001E": "educ25_total",
    "B15003_022E": "educ_bachelors",
    "B15003_023E": "educ_masters",
    "B15003_024E": "educ_professional",
    "B15003_025E": "educ_doctorate",
    "B03002_001E": "race_total",
    "B03002_003E": "nh_white",
    "B03002_004E": "nh_black",
    "B03002_006E": "nh_asian",
    "B03002_012E": "hispanic",
    "B05002_001E": "nativity_total",
    "B05002_013E": "foreign_born",
    "C16002_001E": "lang_households",
    "C16002_004E": "limited_english_spanish",
    "C16002_007E": "limited_english_indo_euro",
    "C16002_010E": "limited_english_asian_pi",
    "C16002_013E": "limited_english_other",
    "B28002_001E": "internet_households",
    "B28002_004E": "broadband_households",
    "B25003_001E": "tenure_total",
    "B25003_002E": "owner_occupied",
    "B25003_003E": "renter_occupied",
    "B01001_001E": "age_total",
    "B01001_020E": "male_65_66",
    "B01001_021E": "male_67_69",
    "B01001_022E": "male_70_74",
    "B01001_023E": "male_75_79",
    "B01001_024E": "male_80_84",
    "B01001_025E": "male_85_plus",
    "B01001_044E": "female_65_66",
    "B01001_045E": "female_67_69",
    "B01001_046E": "female_70_74",
    "B01001_047E": "female_75_79",
    "B01001_048E": "female_80_84",
    "B01001_049E": "female_85_plus",
}


def safe_div(num, den):
    return num.where(den > 0) / den.where(den > 0)


def download(year, api_key=None):
    var_list = list(VARS.keys())
    params = {
        "get": ",".join(var_list),
        "for": "zip code tabulation area:*",
    }
    api_key = api_key or __import__("os").environ.get("CENSUS_API_KEY")
    if api_key:
        params["key"] = api_key
    url = "https://api.census.gov/data/{}/acs/acs5?{}".format(year, urllib.parse.urlencode(params))
    print("requesting", url)
    with urllib.request.urlopen(url, timeout=180) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    header, rows = payload[0], payload[1:]
    df = pd.DataFrame(rows, columns=header)
    df = df.rename(columns=VARS)
    df = df.rename(columns={"zip code tabulation area": "zcta"})
    return df


def derive(df):
    df = df.copy()
    df["zcta"] = df["zcta"].astype(str).str.zfill(5)
    for col in df.columns:
        if col not in {"zcta", "name"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["poverty_rate"] = safe_div(df["poverty_count"], df["poverty_universe"])
    df["unemployment_rate"] = safe_div(df["unemployed_count"], df["civilian_labor_force"])
    df["bachelors_plus_count"] = df[["educ_bachelors", "educ_masters", "educ_professional", "educ_doctorate"]].sum(axis=1)
    df["bachelors_plus_share"] = safe_div(df["bachelors_plus_count"], df["educ25_total"])
    df["nh_white_share"] = safe_div(df["nh_white"], df["race_total"])
    df["nh_black_share"] = safe_div(df["nh_black"], df["race_total"])
    df["nh_asian_share"] = safe_div(df["nh_asian"], df["race_total"])
    df["hispanic_share"] = safe_div(df["hispanic"], df["race_total"])
    df["foreign_born_share"] = safe_div(df["foreign_born"], df["nativity_total"])
    df["limited_english_households"] = df[[
        "limited_english_spanish",
        "limited_english_indo_euro",
        "limited_english_asian_pi",
        "limited_english_other",
    ]].sum(axis=1)
    df["limited_english_household_share"] = safe_div(df["limited_english_households"], df["lang_households"])
    df["broadband_share"] = safe_div(df["broadband_households"], df["internet_households"])
    df["homeownership_share"] = safe_div(df["owner_occupied"], df["tenure_total"])
    df["renter_share"] = safe_div(df["renter_occupied"], df["tenure_total"])
    df["age65_plus_count"] = df[[
        "male_65_66", "male_67_69", "male_70_74", "male_75_79", "male_80_84", "male_85_plus",
        "female_65_66", "female_67_69", "female_70_74", "female_75_79", "female_80_84", "female_85_plus",
    ]].sum(axis=1)
    df["age65_plus_share"] = safe_div(df["age65_plus_count"], df["age_total"])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--out", default="data/processed/acs_zcta_2023.csv")
    parser.add_argument("--api-key", default=None, help="Census API key. Defaults to CENSUS_API_KEY env var.")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = derive(download(args.year, args.api_key))
    df.to_csv(out_path, index=False)
    print("wrote {:,} ZCTAs to {}".format(len(df), out_path))


if __name__ == "__main__":
    main()
