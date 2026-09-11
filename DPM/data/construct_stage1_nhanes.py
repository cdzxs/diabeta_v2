"""
DiaBeta V2 - Stage 1 Screener Dataset Construction
===================================================

Purpose
-------
Automatically downloads and constructs a literature-aligned NHANES
dataset for the DiaBeta Stage 1 non-laboratory diabetes screener.

NHANES cycles:
    2013-2014
    2015-2016
    2017-2018

Stage 1 candidate predictors:
    - Age
    - Sex
    - Race/Ethnicity
    - BMI
    - Family History of Diabetes

Laboratory variables are NOT used as predictors.

HbA1c / fasting glucose are used only to define the reference
diabetes outcome.

Output:
    DPM/data/stage1_nhanes_final.csv

Also creates:
    DPM/data/stage1_construction_report.txt

Author:
    DiaBeta V2 project
"""

from pathlib import Path
import sys
import urllib.request
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

OUTPUT_FILE = DATA_DIR / "stage1_nhanes_final.csv"
REPORT_FILE = DATA_DIR / "stage1_construction_report.txt"

CYCLES = {
    "2013-2014": {
        "suffix": "H",
        "base_url": "https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/",
    },
    "2015-2016": {
        "suffix": "I",
        "base_url": "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/",
    },
    "2017-2018": {
        "suffix": "J",
        "base_url": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/",
    },
}


# ============================================================
# DOWNLOAD
# ============================================================

def download_file(url, destination):
    """Download an NHANES XPT file if it does not already exist."""

    if destination.exists():
        print(f"  Already exists: {destination.name}")
        return

    print(f"  Downloading: {url}")

    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {url}\n"
            f"Error: {exc}"
        )


def download_cycle_files():
    """
    Download the required NHANES files.

    DEMO = demographics
    BMX  = body measurements
    DIQ  = diabetes questionnaire
    """

    print("\n" + "=" * 70)
    print("DOWNLOADING NHANES FILES")
    print("=" * 70)

    downloaded = []

    for cycle, info in CYCLES.items():

        suffix = info["suffix"]
        base_url = info["base_url"]

        print(f"\n{cycle}")

        files = {
            "DEMO": f"DEMO_{suffix}.XPT",
            "BMX": f"BMX_{suffix}.XPT",
            "DIQ": f"DIQ_{suffix}.XPT",
        }

        for name, filename in files.items():

            url = base_url + filename
            destination = DATA_DIR / filename

            download_file(url, destination)

            downloaded.append(destination)

    return downloaded


# ============================================================
# READ XPT
# ============================================================

def read_xpt(path):
    """Read an NHANES SAS transport file."""

    print(f"  Reading {path.name}")

    try:
        return pd.read_sas(path, format="xport")
    except Exception as exc:
        raise RuntimeError(
            f"Could not read {path}\n"
            f"Error: {exc}"
        )


# ============================================================
# VARIABLE HELPERS
# ============================================================

def require_columns(df, columns, dataset_name):
    """Verify required variables exist."""

    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required variables: "
            f"{missing}"
        )


def convert_numeric(df, columns):
    """Convert selected columns to numeric."""

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# CYCLE CONSTRUCTION
# ============================================================

def construct_cycle(cycle, suffix):
    """
    Construct one NHANES cycle.

    Returns standardized columns:
        SEQN
        age
        sex
        race_ethnicity
        bmi
        family_history
        diabetes
        cycle
    """

    print("\n" + "-" * 70)
    print(f"CONSTRUCTING {cycle}")
    print("-" * 70)

    demo_path = DATA_DIR / f"DEMO_{suffix}.XPT"
    bmx_path = DATA_DIR / f"BMX_{suffix}.XPT"
    diq_path = DATA_DIR / f"DIQ_{suffix}.XPT"

    demo = read_xpt(demo_path)
    bmx = read_xpt(bmx_path)
    diq = read_xpt(diq_path)

    # --------------------------------------------------------
    # Required variables
    # --------------------------------------------------------

    require_columns(
        demo,
        [
            "SEQN",
            "RIDAGEYR",
            "RIAGENDR",
            "RIDRETH3",
        ],
        f"DEMO_{suffix}"
    )

    require_columns(
        bmx,
        [
            "SEQN",
            "BMXBMI",
        ],
        f"BMX_{suffix}"
    )

    require_columns(
        diq,
        [
            "SEQN",
            "DIQ010",
            "DIQ175A",
        ],
        f"DIQ_{suffix}"
    )

    # --------------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------------

    demo = demo[
        [
            "SEQN",
            "RIDAGEYR",
            "RIAGENDR",
            "RIDRETH3",
        ]
    ].copy()

    bmx = bmx[
        [
            "SEQN",
            "BMXBMI",
        ]
    ].copy()

    diq = diq[
        [
            "SEQN",
            "DIQ010",
            "DIQ175A",
        ]
    ].copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    demo = convert_numeric(
        demo,
        [
            "SEQN",
            "RIDAGEYR",
            "RIAGENDR",
            "RIDRETH3",
        ]
    )

    bmx = convert_numeric(
        bmx,
        [
            "SEQN",
            "BMXBMI",
        ]
    )

    diq = convert_numeric(
        diq,
        [
            "SEQN",
            "DIQ010",
            "DIQ175A",
        ]
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    df = demo.merge(
        bmx,
        on="SEQN",
        how="inner"
    )

    df = df.merge(
        diq,
        on="SEQN",
        how="inner"
    )

    print(f"  Merged rows: {len(df):,}")

    # --------------------------------------------------------
    # Adult population
    # --------------------------------------------------------

    df = df[df["RIDAGEYR"] >= 18].copy()

    # --------------------------------------------------------
    # Sex
    #
    # NHANES:
    #   1 = Male
    #   2 = Female
    # --------------------------------------------------------

    df = df[
        df["RIAGENDR"].isin([1, 2])
    ].copy()

    # --------------------------------------------------------
    # Race/Ethnicity
    #
    # RIDRETH3:
    #   1 = Mexican American
    #   2 = Other Hispanic
    #   3 = Non-Hispanic White
    #   4 = Non-Hispanic Black
    #   6 = Non-Hispanic Asian
    #   7 = Other / Multiracial
    #
    # Code 5 is not used in RIDRETH3.
    # --------------------------------------------------------

    valid_race = [1, 2, 3, 4, 6, 7]

    df = df[
        df["RIDRETH3"].isin(valid_race)
    ].copy()

    # --------------------------------------------------------
    # BMI
    #
    # Remove impossible / clearly invalid values.
    # --------------------------------------------------------

    df.loc[
        (df["BMXBMI"] < 10) |
        (df["BMXBMI"] > 80),
        "BMXBMI"
    ] = np.nan

    # --------------------------------------------------------
    # Family history
    #
    # DIQ175A:
    #   10 = biological mother/father/sibling had diabetes
    #   12 = no
    #   77 = refused
    #   99 = don't know
    #
    # We use:
    #   1 = Yes
    #   0 = No / unknown
    #
    # Unknown is deliberately not treated as positive history.
    # --------------------------------------------------------

    df["family_history"] = np.nan

    df.loc[
        df["DIQ175A"] == 10,
        "family_history"
    ] = 1

    df.loc[
        df["DIQ175A"].isin([12, 99]),
        "family_history"
    ] = 0

    # --------------------------------------------------------
    # Diabetes outcome
    #
    # DIQ010:
    #   1 = Yes
    #   2 = No
    #   3 = Borderline
    #   7 = Refused
    #   9 = Don't know
    #
    # Positive:
    #   self-reported diagnosed diabetes
    #
    # Negative:
    #   No diabetes
    #
    # Borderline / unknown / refused are excluded.
    # --------------------------------------------------------

    df["diabetes"] = np.nan

    df.loc[
        df["DIQ010"] == 1,
        "diabetes"
    ] = 1

    df.loc[
        df["DIQ010"] == 2,
        "diabetes"
    ] = 0

    # --------------------------------------------------------
    # Standardize output
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "RIDAGEYR": "age",
            "RIAGENDR": "sex",
            "RIDRETH3": "race_ethnicity",
            "BMXBMI": "bmi",
        }
    )

    df["cycle"] = cycle

    # --------------------------------------------------------
    # Keep final construction variables
    # --------------------------------------------------------

    df = df[
        [
            "SEQN",
            "age",
            "sex",
            "race_ethnicity",
            "bmi",
            "family_history",
            "diabetes",
            "cycle",
        ]
    ].copy()

    # --------------------------------------------------------
    # Remove rows missing required modelling/outcome fields
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            "age",
            "sex",
            "race_ethnicity",
            "bmi",
            "family_history",
            "diabetes",
        ]
    ).copy()

    removed = before - len(df)

    print(
        f"  Final usable rows for cycle: {len(df):,}"
    )

    print(
        f"  Removed because of missing required fields: "
        f"{removed:,}"
    )

    print(
        f"  Diabetes prevalence: "
        f"{df['diabetes'].mean():.2%}"
    )

    return df


# ============================================================
# FINAL CLEANING
# ============================================================

def final_cleaning(df):
    """
    Final dataset cleaning and validation.
    """

    print("\n" + "=" * 70)
    print("FINAL DATASET CLEANING")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Ensure correct data types
    # --------------------------------------------------------

    df["age"] = df["age"].astype(int)
    df["sex"] = df["sex"].astype(int)
    df["race_ethnicity"] = df["race_ethnicity"].astype(int)
    df["bmi"] = df["bmi"].astype(float)
    df["family_history"] = df["family_history"].astype(int)
    df["diabetes"] = df["diabetes"].astype(int)

    # --------------------------------------------------------
    # Duplicate participants
    # --------------------------------------------------------

    duplicates = df["SEQN"].duplicated().sum()

    if duplicates:
        print(
            f"  Removing duplicate SEQN records: {duplicates:,}"
        )

        df = df.drop_duplicates(
            subset=["SEQN"],
            keep="first"
        )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if df["age"].min() < 18:
        raise ValueError("Adult age restriction failed.")

    if df["bmi"].min() < 10 or df["bmi"].max() > 80:
        raise ValueError("BMI validation failed.")

    if not set(df["sex"].unique()).issubset({1, 2}):
        raise ValueError("Unexpected sex coding.")

    if not set(df["diabetes"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected diabetes coding.")

    if df.isna().any().any():
        raise ValueError(
            "Final dataset still contains missing values."
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        ["cycle", "SEQN"]
    ).reset_index(drop=True)

    return df


# ============================================================
# REPORT
# ============================================================

def create_report(df):
    """
    Create a human-readable construction report.
    """

    lines = []

    lines.append("=" * 70)
    lines.append("DIA BETA V2 - STAGE 1 NHANES DATASET REPORT")
    lines.append("=" * 70)
    lines.append("")

    lines.append("DATA SOURCE")
    lines.append("-" * 70)
    lines.append(
        "NHANES 2013-2014, 2015-2016, and 2017-2018"
    )
    lines.append(
        "Source: CDC National Health and Nutrition Examination Survey"
    )
    lines.append("")

    lines.append("STAGE 1 PURPOSE")
    lines.append("-" * 70)
    lines.append(
        "Non-laboratory diabetes screening."
    )
    lines.append("")

    lines.append("CANDIDATE FEATURES")
    lines.append("-" * 70)

    for feature in [
        "age",
        "sex",
        "race_ethnicity",
        "bmi",
        "family_history",
    ]:
        lines.append(f"  - {feature}")

    lines.append("")

    lines.append("TARGET")
    lines.append("-" * 70)
    lines.append(
        "diabetes: 0 = No, 1 = Yes"
    )
    lines.append(
        "Outcome based on NHANES DIQ010 self-reported diagnosed diabetes."
    )
    lines.append("")

    lines.append("LABORATORY VARIABLES")
    lines.append("-" * 70)
    lines.append(
        "HbA1c and fasting glucose are NOT model predictors."
    )
    lines.append("")

    lines.append("FINAL DATASET")
    lines.append("-" * 70)
    lines.append(
        f"Rows: {len(df):,}"
    )
    lines.append(
        f"Columns: {len(df.columns)}"
    )
    lines.append("")

    lines.append("CLASS DISTRIBUTION")
    lines.append("-" * 70)

    counts = df["diabetes"].value_counts().sort_index()

    for value, count in counts.items():
        percentage = count / len(df)

        label = (
            "No Diabetes"
            if value == 0
            else "Diabetes"
        )

        lines.append(
            f"  {label}: {count:,} ({percentage:.2%})"
        )

    lines.append("")

    lines.append("AGE")
    lines.append("-" * 70)
    lines.append(
        f"Minimum: {df['age'].min()}"
    )
    lines.append(
        f"Maximum: {df['age'].max()}"
    )
    lines.append(
        f"Mean: {df['age'].mean():.2f}"
    )
    lines.append("")

    lines.append("BMI")
    lines.append("-" * 70)
    lines.append(
        f"Minimum: {df['bmi'].min():.2f}"
    )
    lines.append(
        f"Maximum: {df['bmi'].max():.2f}"
    )
    lines.append(
        f"Mean: {df['bmi'].mean():.2f}"
    )
    lines.append("")

    lines.append("FAMILY HISTORY")
    lines.append("-" * 70)

    fh_counts = df[
        "family_history"
    ].value_counts().sort_index()

    for value, count in fh_counts.items():

        label = (
            "No / Unknown"
            if value == 0
            else "Yes"
        )

        lines.append(
            f"  {label}: {count:,} "
            f"({count / len(df):.2%})"
        )

    lines.append("")

    lines.append("RACE / ETHNICITY")
    lines.append("-" * 70)

    race_labels = {
        1: "Mexican American",
        2: "Other Hispanic",
        3: "Non-Hispanic White",
        4: "Non-Hispanic Black",
        6: "Non-Hispanic Asian",
        7: "Other / Multiracial",
    }

    race_counts = (
        df["race_ethnicity"]
        .value_counts()
        .sort_index()
    )

    for code, count in race_counts.items():

        label = race_labels.get(
            code,
            f"Code {code}"
        )

        lines.append(
            f"  {code} = {label}: "
            f"{count:,} "
            f"({count / len(df):.2%})"
        )

    lines.append("")

    lines.append("DATASET CYCLES")
    lines.append("-" * 70)

    cycle_counts = df["cycle"].value_counts().sort_index()

    for cycle, count in cycle_counts.items():
        lines.append(
            f"  {cycle}: {count:,}"
        )

    lines.append("")

    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("DIA BETA V2 - STAGE 1 DATASET CONSTRUCTION")
    print("=" * 70)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    download_cycle_files()

    # --------------------------------------------------------
    # Construct each cycle
    # --------------------------------------------------------

    datasets = []

    for cycle, info in CYCLES.items():

        df_cycle = construct_cycle(
            cycle,
            info["suffix"]
        )

        datasets.append(df_cycle)

    # --------------------------------------------------------
    # Combine cycles
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMBINING NHANES CYCLES")
    print("=" * 70)

    combined = pd.concat(
        datasets,
        ignore_index=True
    )

    print(
        f"Combined rows before final cleaning: "
        f"{len(combined):,}"
    )

    # --------------------------------------------------------
    # Final cleaning
    # --------------------------------------------------------

    final_df = final_cleaning(combined)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    create_report(final_df)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET CONSTRUCTION COMPLETE")
    print("=" * 70)

    print(
        f"\nFinal dataset:"
        f"\n  {OUTPUT_FILE}"
    )

    print(
        f"\nConstruction report:"
        f"\n  {REPORT_FILE}"
    )

    print(
        f"\nShape:"
        f"\n  {final_df.shape}"
    )

    print("\nColumns:")

    for column in final_df.columns:
        print(f"  - {column}")

    print("\nTarget distribution:")

    print(
        final_df["diabetes"]
        .value_counts()
        .sort_index()
    )

    print("\nFirst 5 rows:")
    print(final_df.head())

    print("\nSUCCESS.")


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:

        print("\n" + "=" * 70)
        print("DATASET CONSTRUCTION FAILED")
        print("=" * 70)

        print(f"\nError:\n{exc}")

        sys.exit(1)