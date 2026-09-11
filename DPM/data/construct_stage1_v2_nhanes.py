"""
DiaBeta V2 - Stage 1 Screener Dataset Construction (8-predictor, v2)
====================================================================

Builds the locked 8-predictor Stage 1 screener dataset from NHANES
2013-2014, 2015-2016, and 2017-2018.

STAGE 1 PREDICTORS (locked, non-laboratory only):
    1. age                (RIDAGEYR)
    2. sex                (RIAGENDR)
    3. bmi                (BMXBMI)
    4. race_ethnicity     (RIDRETH3)
    5. family_history     (MCQ300C)
    6. hypertension       (BPQ020)
    7. physical_activity  (PAQ650 + PAQ665)
    8. smoking_status     (SMQ020 + SMQ040)

HbA1c (LBXGH) and fasting glucose (LBXGLU) are NEVER predictors.
They are used ONLY to construct the 3-class clinical glycemic-status
training target alongside the self-reported diabetes diagnosis
(DIQ010):

    HIGH      = DIQ010 == 1  OR  HbA1c >= 6.5  OR  fasting glucose >= 126
    MODERATE  = not High AND (DIQ010 == 3  OR  5.7 <= HbA1c <= 6.4
                              OR  100 <= fasting glucose <= 125)
    LOW       = not High/Moderate AND sufficient evidence (DIQ010 == 2
                or at least one valid lab value)

Either lab may be used independently - fasting glucose is a subsample,
so requiring both would destroy sample size. Participants whose target
cannot be determined (no diagnosis answer and no valid lab) are EXCLUDED.

Cohort: NHANES adults age >= 20. Ages 18-19 are excluded because the
locked family-history variable MCQ300C is not asked for that range.

DATA QUALITY:
    * every downloaded file is validated as a real SAS XPT (not HTML)
    * required variables are verified to exist before proceeding
    * row counts are reported after every major merge
    * missingness is reported for all 8 predictors and the target
    * class distribution is reported
    * NO outlier removal, NO SMOTE, NO scaling/encoding, NO training

Outputs (new filenames - nothing existing is overwritten):
    DPM/data/stage1_v2_nhanes_8predictors.csv
    DPM/data/stage1_v2_construction_report.txt
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

OUTPUT_FILE = DATA_DIR / "stage1_v2_nhanes_8predictors.csv"
REPORT_FILE = DATA_DIR / "stage1_v2_construction_report.txt"

# CDC now serves NHANES XPT files from this path layout:
#   https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{FILE}.xpt
CYCLES = {
    "2013-2014": {"suffix": "H", "year": 2013},
    "2015-2016": {"suffix": "I", "year": 2015},
    "2017-2018": {"suffix": "J", "year": 2017},
}

# Files needed per cycle and the variables required from each.
REQUIRED_FILES = {
    "DEMO": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"],
    "BMX":  ["SEQN", "BMXBMI"],
    "DIQ":  ["SEQN", "DIQ010"],
    "MCQ":  ["SEQN", "MCQ300C"],
    "BPQ":  ["SEQN", "BPQ020"],
    "PAQ":  ["SEQN", "PAQ650", "PAQ665"],
    "SMQ":  ["SEQN", "SMQ020", "SMQ040"],
    "GHB":  ["SEQN", "LBXGH"],
    "GLU":  ["SEQN", "LBXGLU"],
}

VALID_RACE = [1, 2, 3, 4, 6, 7]
ADULT_MIN_AGE = 20


# ============================================================
# DOWNLOAD + VALIDATE
# ============================================================

def download_file(url, destination):
    """Download an NHANES XPT file if it does not already exist.
    Raises if the response is HTML / an error page, not a real XPT."""

    if destination.exists():
        print(f"  Already exists: {destination.name}")
        return

    print(f"  Downloading: {url}")

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DiaBeta-V2-dataset-builder)"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            content_type = response.headers.get("Content-Type", "")
            if "html" in content_type.lower():
                raise RuntimeError(
                    f"Server returned HTML (Content-Type: {content_type}) for {url} - "
                    f"refusing to save it as an XPT file."
                )
            raw = response.read()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to download {url}\nError: {exc}")

    if len(raw) < 100:
        raise RuntimeError(
            f"Suspiciously small file ({len(raw)} bytes) from {url} - not a valid XPT."
        )

    # SAS XPORT files begin with a fixed "HEADER RECORD" marker.
    if not raw.startswith(b"HEADER"):
        preview = raw[:60]
        raise RuntimeError(
            f"File from {url} is not a SAS XPT file (missing HEADER magic). "
            f"First bytes: {preview!r}"
        )

    destination.write_bytes(raw)
    print(f"  Saved: {destination.name} ({len(raw):,} bytes)")


def download_cycle_files(cycle_info):
    """Download the nine required files for one cycle."""

    print("\n" + "=" * 70)
    print(f"DOWNLOADING NHANES FILES - {cycle_info['label']}")
    print("=" * 70)

    base_url = (
        f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/"
        f"{cycle_info['year']}/DataFiles/"
    )

    for stem in REQUIRED_FILES:
        filename = f"{stem}_{cycle_info['suffix']}.xpt"
        download_file(base_url + filename, DATA_DIR / filename)


# ============================================================
# READ + VALIDATE VARIABLES
# ============================================================

def read_xpt(path):
    """Read a validated NHANES SAS transport file; uppercase column names."""

    print(f"  Reading {path.name}")

    df = pd.read_sas(path, format="xport")
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def require_columns(df, columns, dataset_name):
    """Verify required variables exist before proceeding."""

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required variables: {missing}"
        )
    return True


def load_validated(path, required, dataset_name):
    """Read a file and verify it has every required variable."""

    df = read_xpt(path)
    require_columns(df, required, dataset_name)
    return df


# ============================================================
# PREDICTOR DERIVATION
# ============================================================

def derive_family_history(s):
    """MCQ300C: 1 = Yes, 2 = No, 7/9 = refused/don't know -> missing."""

    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[s == 1] = 1.0
    out[s == 2] = 0.0
    return out


def derive_hypertension(s):
    """BPQ020: 1 = Yes, 2 = No, 7/9 = refused/don't know -> missing."""

    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[s == 1] = 1.0
    out[s == 2] = 0.0
    return out


def derive_physical_activity(vig, mod):
    """PAQ650 (vigorous rec.) / PAQ665 (moderate rec.): 1 = Yes, 2 = No.
    1 if either activity is Yes; 0 only if BOTH are explicitly No;
    missing if the information is insufficient/unknown."""

    out = pd.Series(np.nan, index=vig.index, dtype=float)
    out[(vig == 1) | (mod == 1)] = 1.0
    out[(vig == 2) & (mod == 2)] = 0.0
    return out


def derive_smoking_status(smq020, smq040):
    """SMQ020 (>=100 cigarettes lifetime) + SMQ040 (current frequency).
    Never: SMQ020 == 2 (SMQ040 skipped for never-smokers - NOT missing).
    Former: SMQ020 == 1 and SMQ040 == 3 (not at all now).
    Current: SMQ020 == 1 and SMQ040 in (1, 2).
    Missing: unknown SMQ020, or current-smoking frequency unknown/refused."""

    out = pd.Series(np.nan, index=smq020.index, dtype=float)

    never = smq020 == 2
    out[never] = 0.0

    former = (smq020 == 1) & (smq040 == 3)
    out[former] = 1.0

    current = (smq020 == 1) & (smq040.isin([1, 2]))
    out[current] = 2.0

    return out


# ============================================================
# TARGET CONSTRUCTION
# ============================================================

def derive_target(diq010, hba1c, glucose):
    """3-class clinical glycemic-status target.
    Returns (label series, undetermined mask); undetermined participants
    are excluded from the supervised dataset (never assigned arbitrarily)."""

    high = (
        (diq010 == 1)
        | (hba1c >= 6.5)
        | (glucose >= 126)
    )

    moderate = (
        ~high
        & (
            (diq010 == 3)
            | ((hba1c >= 5.7) & (hba1c <= 6.4))
            | ((glucose >= 100) & (glucose <= 125))
        )
    )

    # Low requires sufficient evidence: a "No" diagnosis (DIQ010 == 2)
    # OR at least one valid laboratory measurement.
    has_lab = hba1c.notna() | glucose.notna()
    low = (
        ~high
        & ~moderate
        & ((diq010 == 2) | has_lab)
    )

    labels = pd.Series(np.nan, index=diq010.index, dtype=object)
    labels[high] = "High"
    labels[moderate] = "Moderate"
    labels[low] = "Low"

    undetermined = labels.isna()
    return labels, undetermined


# ============================================================
# CYCLE CONSTRUCTION
# ============================================================

def construct_cycle(cycle, suffix, year):
    """Build one NHANES cycle into the standardized v2 schema."""

    print("\n" + "-" * 70)
    print(f"CONSTRUCTING {cycle}")
    print("-" * 70)

    def path_for(stem):
        return DATA_DIR / f"{stem}_{suffix}.xpt"

    # ---- Read all nine files (with variable validation) ----
    demo = load_validated(path_for("DEMO"), REQUIRED_FILES["DEMO"], f"DEMO_{suffix}")
    bmx  = load_validated(path_for("BMX"),  REQUIRED_FILES["BMX"],  f"BMX_{suffix}")
    diq  = load_validated(path_for("DIQ"),  REQUIRED_FILES["DIQ"],  f"DIQ_{suffix}")
    mcq  = load_validated(path_for("MCQ"),  REQUIRED_FILES["MCQ"],  f"MCQ_{suffix}")
    bpq  = load_validated(path_for("BPQ"),  REQUIRED_FILES["BPQ"],  f"BPQ_{suffix}")
    paq  = load_validated(path_for("PAQ"),  REQUIRED_FILES["PAQ"],  f"PAQ_{suffix}")
    smq  = load_validated(path_for("SMQ"),  REQUIRED_FILES["SMQ"],  f"SMQ_{suffix}")
    ghb  = load_validated(path_for("GHB"),  REQUIRED_FILES["GHB"],  f"GHB_{suffix}")
    glu  = load_validated(path_for("GLU"),  REQUIRED_FILES["GLU"],  f"GLU_{suffix}")

    print(f"  Raw rows: DEMO={len(demo):,} BMX={len(bmx):,} DIQ={len(diq):,} "
          f"MCQ={len(mcq):,} BPQ={len(bpq):,} PAQ={len(paq):,} SMQ={len(smq):,} "
          f"GHB={len(ghb):,} GLU={len(glu):,}")

    # ---- Core merge: demographics + exam + diabetes questionnaire ----
    core = demo.merge(bmx, on="SEQN", how="inner")
    print(f"  After DEMO+BMX merge: {len(core):,}")
    core = core.merge(diq, on="SEQN", how="inner")
    print(f"  After +DIQ merge: {len(core):,}")

    # ---- Questionnaire files (MCQ, BPQ, PAQ, SMQ) ----
    for stem, df in [("MCQ", mcq), ("BPQ", bpq), ("PAQ", paq), ("SMQ", smq)]:
        keep = [c for c in REQUIRED_FILES[stem]]
        core = core.merge(df[keep], on="SEQN", how="inner")
        print(f"  After +{stem} merge: {len(core):,}")

    # ---- Lab files are PARTIAL by design (GHB ~full exam, GLU fasting
    #      subsample). LEFT merge so participants without a lab remain -
    #      the target can still be set from DIQ010 alone. ----
    core = core.merge(ghb[["SEQN", "LBXGH"]], on="SEQN", how="left")
    print(f"  After +GHB (HbA1c, left): {len(core):,}")
    core = core.merge(glu[["SEQN", "LBXGLU"]], on="SEQN", how="left")
    print(f"  After +GLU (fasting glucose, left): {len(core):,}")

    # ---- Cohort: adults >= 20 ----
    # RIDAGEYR can carry a tiny SAS-missing artifact (~1e-79); the >= 20
    # filter removes both the 18-19s and any non-valid age codes.
    before_age = len(core)
    core = core[core["RIDAGEYR"] >= ADULT_MIN_AGE].copy()
    print(f"  Age >= 20 filter: {before_age:,} -> {len(core):,}")

    # ---- Valid sex / race coding ----
    core = core[core["RIAGENDR"].isin([1, 2])].copy()
    core = core[core["RIDRETH3"].isin(VALID_RACE)].copy()

    # ---- Numeric conversion ----
    for col in ["RIDAGEYR", "RIAGENDR", "RIDRETH3", "BMXBMI",
                "MCQ300C", "BPQ020", "PAQ650", "PAQ665",
                "SMQ020", "SMQ040", "DIQ010", "LBXGH", "LBXGLU"]:
        core[col] = pd.to_numeric(core[col], errors="coerce")

    # ---- Derive the eight predictors ----
    out = pd.DataFrame({"SEQN": core["SEQN"]})
    out["age"] = core["RIDAGEYR"].astype(int)
    out["sex"] = core["RIAGENDR"].astype(int)
    out["bmi"] = core["BMXBMI"]
    out["race_ethnicity"] = core["RIDRETH3"].astype(int)
    out["family_history"] = derive_family_history(core["MCQ300C"])
    out["hypertension"] = derive_hypertension(core["BPQ020"])
    out["physical_activity"] = derive_physical_activity(core["PAQ650"], core["PAQ665"])
    out["smoking_status"] = derive_smoking_status(core["SMQ020"], core["SMQ040"])

    # ---- Target (raw DIQ010 kept for audit provenance; labs NOT saved) ----
    target, undetermined = derive_target(core["DIQ010"], core["LBXGH"], core["LBXGLU"])
    out["diq010"] = core["DIQ010"]
    out["diabetes_risk"] = target
    out["_undetermined"] = undetermined

    # ---- Lab availability (for the report) ----
    out["_has_hba1c"] = core["LBXGH"].notna()
    out["_has_glucose"] = core["LBXGLU"].notna()

    out["cycle"] = cycle

    print(
        f"  Cycle rows: {len(out):,} | "
        f"target determined: {int((~undetermined).sum()):,} | "
        f"undetermined: {int(undetermined.sum()):,}"
    )

    return out


# ============================================================
# FINAL CLEANING
# ============================================================

PREDICTOR_ORDER = [
    "age", "sex", "bmi", "race_ethnicity",
    "family_history", "hypertension", "physical_activity", "smoking_status",
]


def final_cleaning(df):
    """Combine cycles, remove undetermined targets, sanity checks."""

    print("\n" + "=" * 70)
    print("FINAL CLEANING")
    print("=" * 70)

    df = df.copy()

    # Drop the temporary audit columns
    has_hba1c = df.pop("_has_hba1c")
    has_glucose = df.pop("_has_glucose")
    cohort_n = len(df)

    # ---- Cohort-level missingness (BEFORE target filtering) ----
    cohort_missing = {}
    for p in PREDICTOR_ORDER:
        n_miss = int(df[p].isna().sum())
        cohort_missing[p] = (n_miss, n_miss / cohort_n)
    cohort_missing["diabetes_risk"] = int(df["_undetermined"].sum())

    print(
        f"Combined rows before target filtering: {len(df):,}"
    )

    # ---- Exclude undetermined targets ----
    excluded_mask = df.pop("_undetermined")
    n_undetermined = int(excluded_mask.sum())
    print(
        f"Participants excluded (target undetermined): {n_undetermined:,}"
    )

    df = df[~excluded_mask].copy()

    # ---- Type casts ----
    df["age"] = df["age"].astype(int)
    df["sex"] = df["sex"].astype(int)
    df["race_ethnicity"] = df["race_ethnicity"].astype(int)
    df["bmi"] = df["bmi"].astype(float)
    df["diq010"] = pd.to_numeric(df["diq010"], errors="coerce")

    # ---- Sanity checks ----
    if df["age"].min() < ADULT_MIN_AGE:
        raise ValueError("Adult age restriction failed.")
    if not set(df["sex"].unique()).issubset({1, 2}):
        raise ValueError("Unexpected sex coding.")
    if not set(df["race_ethnicity"].unique()).issubset(set(VALID_RACE)):
        raise ValueError("Unexpected race/ethnicity coding.")
    if not set(df["diabetes_risk"].unique()).issubset({"Low", "Moderate", "High"}):
        raise ValueError("Unexpected target values.")

    # ---- No leakage: labs must NEVER be part of the saved dataset ----
    labish = [c for c in df.columns if c.lower() in
              ("hba1c", "lbxgh", "glucose", "fasting_glucose", "lbxglu")]
    if labish:
        raise ValueError(f"Lab columns leaked into output: {labish}")

    # ---- Deterministic order ----
    final_cols = ["SEQN", "cycle"] + PREDICTOR_ORDER + ["diq010", "diabetes_risk"]
    df = df[final_cols].sort_values(["cycle", "SEQN"]).reset_index(drop=True)

    print(f"Final supervised rows: {len(df):,}")
    return df, n_undetermined, has_hba1c, has_glucose, cohort_n, cohort_missing


# ============================================================
# REPORT
# ============================================================

def create_report(df, n_excluded_undetermined, has_hba1c, has_glucose,
                  cohort_n, cohort_missing):
    """Write a human-readable construction report and return its text."""

    lines = []
    def out(text=""):
        print(text)
        lines.append(text)

    out("=" * 70)
    out("DIA BETA V2 - STAGE 1 (8-PREDICTOR) NHANES DATASET REPORT")
    out("=" * 70)
    out()
    out("DATA SOURCE")
    out("-" * 70)
    out("NHANES 2013-2014, 2015-2016, 2017-2018 (CDC)")
    out("Files: DEMO, BMX, DIQ, MCQ, BPQ, PAQ, SMQ, GHB, GLU")
    out()
    out("COHORT")
    out("-" * 70)
    out("Adults age >= 20 (18-19 excluded: MCQ300C family-history")
    out("variable is not asked for that age range).")
    out()
    out("PREDICTORS (8, non-laboratory only)")
    out("-" * 70)
    for p in ["age", "sex", "bmi", "race_ethnicity",
              "family_history", "hypertension", "physical_activity",
              "smoking_status"]:
        out(f"  - {p}")
    out()
    out("LABORATORY VARIABLES")
    out("-" * 70)
    out("HbA1c (LBXGH) and fasting glucose (LBXGLU) are NOT predictors.")
    out("They are used ONLY to construct the training target.")
    out("No laboratory column is saved in the dataset.")
    out()
    out("TARGET (diabetes_risk, 3-class)")
    out("-" * 70)
    out("  High     = DIQ010==1 OR HbA1c>=6.5 OR fasting glucose>=126")
    out("  Moderate = not High AND (DIQ010==3 OR 5.7<=HbA1c<=6.4 OR 100<=FG<=125)")
    out("  Low      = not High/Moderate AND (DIQ010==2 OR >=1 valid lab)")
    out("Either lab is used independently (fasting glucose is a subsample).")
    out("Undetermined targets are excluded, never assigned arbitrarily.")
    out()
    out("FINAL DATASET")
    out("-" * 70)
    out(f"  Rows: {len(df):,}")
    out(f"  Predictors: 8")
    out(f"  Columns: {len(df.columns)}")
    out()
    out("TARGET CLASS DISTRIBUTION")
    out("-" * 70)
    counts = df["diabetes_risk"].value_counts().reindex(
        ["Low", "Moderate", "High"]
    ).fillna(0).astype(int)
    for cls in ["Low", "Moderate", "High"]:
        n = counts[cls]
        out(f"  {cls}: {n:,} ({n / len(df):.2%})")
    out()
    out("MISSINGNESS - FINAL SUPERVISED DATASET")
    out("-" * 70)
    for p in PREDICTOR_ORDER:
        n_miss = int(df[p].isna().sum())
        out(f"  {p}: {n_miss:,} ({n_miss / len(df):.2%})")
    out()
    out("MISSINGNESS - COHORT (before target filtering)")
    out("-" * 70)
    out(f"  Cohort rows: {cohort_n:,}")
    for p in PREDICTOR_ORDER:
        n_miss, pct = cohort_missing[p]
        out(f"  {p}: {n_miss:,} ({pct:.2%})")
    out(
        f"  diabetes_risk (undetermined): "
        f"{cohort_missing['diabetes_risk']:,} "
        f"({cohort_missing['diabetes_risk'] / cohort_n:.2%})"
    )
    out(f"  HbA1c available:   {int(has_hba1c.sum()):,} "
        f"({has_hba1c.mean():.1%})")
    out(f"  Fasting glucose available: {int(has_glucose.sum()):,} "
        f"({has_glucose.mean():.1%})")
    out(f"  Both labs: {int((has_hba1c & has_glucose).sum()):,} | "
        f"Neither lab: {int((~has_hba1c & ~has_glucose).sum()):,}")
    out()
    out("PARTICIPANTS EXCLUDED")
    out("-" * 70)
    out(f"  Target undetermined (no diagnosis answer AND no valid lab): "
        f"{n_excluded_undetermined:,}")
    out("  (Age 18-19 and non-adults are filtered inside each cycle;")
    out("   invalid sex/race codes are filtered as data hygiene.)")
    out()
    out("CYCLES REPRESENTED")
    out("-" * 70)
    cycle_counts = df["cycle"].value_counts().sort_index()
    for cycle, n in cycle_counts.items():
        out(f"  {cycle}: {n:,}")
    out()
    out("=" * 70)
    out("END OF REPORT")
    out("=" * 70)

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("DIA BETA V2 - STAGE 1 DATASET CONSTRUCTION (8 PREDICTORS)")
    print("=" * 70)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Download every cycle ----
    datasets = []
    for label, info in CYCLES.items():
        cycle_info = dict(label=label, suffix=info["suffix"], year=info["year"])
        download_cycle_files(cycle_info)
        datasets.append(construct_cycle(label, info["suffix"], info["year"]))

    # ---- Combine ----
    print("\n" + "=" * 70)
    print("COMBINING NHANES CYCLES")
    print("=" * 70)

    combined = pd.concat(datasets, ignore_index=True)
    print(f"Combined rows: {len(combined):,}")

    final_df, n_undetermined, has_hba1c, has_glucose, cohort_n, cohort_missing = final_cleaning(combined)

    # ---- Save ----
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")

    create_report(final_df, n_undetermined, has_hba1c, has_glucose,
                  cohort_n, cohort_missing)
    print(f"\nSaved: {REPORT_FILE}")

    print("\n" + "=" * 70)
    print("DATASET CONSTRUCTION COMPLETE")
    print("=" * 70)
    print(f"\nShape: {final_df.shape}")
    print("\nTarget distribution:")
    print(final_df["diabetes_risk"].value_counts().reindex(
        ["Low", "Moderate", "High"]).fillna(0))
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