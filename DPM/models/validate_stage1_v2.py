"""
DiaBeta V2 - Stage 1 V2 Candidate Behavioral Validation
========================================================

VALIDATION ONLY. No model artifact is created or modified, and nothing is
integrated. Loads:

    DPM/models/stage1_v2_candidate.pkl     (improved Stage 1 V2 candidate)
    DPM/models/diabeta_dataset1_model.pkl  (old frozen Model 1B - load-only)

and runs a behavioral sanity suite:

    PART 1  artifact + pipeline + predictor-interface verification
    PART 2  synthetic low / moderate / high profiles (+ combos)
    PART 3  controlled grid (age x BMI x family_history x hypertension)
    PART 4  trend checks (age, BMI, family history, hypertension)
    PART 5  old vs new candidate comparison
    PART 6  PASS/FAIL decision

Output (text only):
    DPM/models/stage1_v2_validation_report.txt
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent          # .../DiaBeta/DPM
CANDIDATE = BASE_DIR / "models" / "stage1_v2_candidate.pkl"
OLD_MODEL = BASE_DIR / "models" / "diabeta_dataset1_model.pkl"
REPORT_OUT = BASE_DIR / "models" / "stage1_v2_validation_report.txt"

PREDICTORS = [
    "age", "sex", "bmi", "race_ethnicity",
    "family_history", "hypertension", "physical_activity", "smoking_status",
]
FORBIDDEN = ["hba1c", "fasting_glucose", "diq010", "SEQN", "cycle"]

# Candidate was trained with LabelEncoder -> classes sorted alphabetically.
LE_CLASSES = ["High", "Low", "Moderate"]   # index 0/1/2


def enc_to_label(code):
    return LE_CLASSES[int(code)]


LINES = []


def out(text=""):
    print(text)
    LINES.append(text)


def predict_candidate(pipe, profiles_df):
    """Return (labels, proba_df with P(Low)/P(Moderate)/P(High))."""
    codes = pipe.predict(profiles_df)
    proba = pipe.predict_proba(profiles_df)
    proba_df = pd.DataFrame(proba, columns=LE_CLASSES)
    proba_df = proba_df[["Low", "Moderate", "High"]]
    labels = [enc_to_label(c) for c in codes]
    return labels, proba_df


def profile_row(d):
    return {p: d.get(p) for p in PREDICTORS}


# ============================================================
# MAIN
# ============================================================

def main():
    out("=" * 70)
    out("DIA BETA V2 - STAGE 1 V2 CANDIDATE BEHAVIORAL VALIDATION")
    out("=" * 70)

    # ---------------- PART 1: artifact verification ----------------
    out("\n" + "=" * 70)
    out("PART 1 - ARTIFACT VERIFICATION")
    out("=" * 70)

    pipe = joblib.load(CANDIDATE)
    out(f"1. Loaded: {CANDIDATE.name} -> OK")

    out(f"2. Is sklearn Pipeline: {isinstance(pipe, __import__('sklearn').pipeline.Pipeline)}")
    clf = pipe.named_steps["clf"]
    out(f"3. Final estimator: {type(clf).__name__}")

    params = clf.get_params()
    for k in ["max_iter", "max_leaf_nodes", "min_samples_leaf",
              "l2_regularization", "random_state"]:
        out(f"   {k}: {params[k]}")

    out("4. Preprocessing:")
    prep = pipe.named_steps["prep"]
    for name, tr, cols in prep.transformers:
        steps = " -> ".join(s[0] for s in tr.steps) if hasattr(tr, "steps") else str(tr)
        out(f"   [{name}] columns={list(cols)}  steps: {steps}")

    fni = list(prep.feature_names_in_) if hasattr(prep, "feature_names_in_") else None
    out(f"5. Expected predictor columns (feature_names_in_): {fni}")
    out(f"   Exactly the 8 locked predictors: {fni == PREDICTORS}")
    overlap = [f for f in (fni or []) if f in FORBIDDEN]
    out(f"6. Forbidden predictors present (HbA1c/glucose/diq010/SEQN/cycle): "
        f"{overlap if overlap else 'NONE'}")

    # ---------------- PART 2: synthetic profiles ----------------
    out("\n" + "=" * 70)
    out("PART 2 - PREDICTION BEHAVIOR (synthetic profiles)")
    out("=" * 70)

    def base(age, bmi, fh, htn, sex=1, race=1, pa=1, smoke=0):
        return dict(age=age, sex=sex, bmi=bmi, race_ethnicity=race,
                    family_history=fh, hypertension=htn,
                    physical_activity=pa, smoking_status=smoke)

    low_profiles = [
        ("A1", base(20, 20, 0, 0)),
        ("A2", base(25, 22, 0, 0, sex=2, race=2)),
        ("A3", base(30, 24, 0, 0, race=3)),
    ]
    mod_profiles = [
        ("B1", base(40, 28, 1, 0)),
        ("B2", base(50, 30, 1, 1)),
        ("B3", base(55, 32, 0, 1)),
        ("B4", base(45, 31, 1, 1)),
    ]
    high_profiles = [
        ("C1", base(60, 35, 1, 1)),
        ("C2", base(65, 40, 1, 1)),
        ("C3", base(70, 42, 1, 1)),
        ("C4", base(75, 45, 1, 1)),
    ]
    combo_profiles = [
        ("D1 female/race4/smoke2", base(58, 33, 1, 1, sex=2, race=4, smoke=2)),
        ("D2 female/race6/smoke0", base(48, 29, 1, 0, sex=2, race=6, smoke=0)),
        ("D3 male/race7/pa0", base(52, 31, 0, 1, race=7, pa=0)),
        ("D4 male/race2/pa0/smoke1", base(44, 27, 1, 0, race=2, pa=0, smoke=1)),
        ("D5 female/race3/smoke1", base(62, 36, 1, 1, sex=2, race=3, smoke=1)),
        ("D6 male/race1/pa0/smoke2", base(38, 26, 0, 0, pa=0, smoke=2)),
    ]

    def run_profiles(title, profiles):
        out(f"\n  {title}")
        out(f"  {'id':<4}{'age':>4}{'sex':>4}{'bmi':>5}{'race':>5}{'fh':>4}"
            f"{'htn':>4}{'pa':>4}{'smk':>4}  {'pred':>8}"
            f"{'P(Low)':>9}{'P(Mod)':>9}{'P(High)':>9}")
        rows = [profile_row(d) for _, d in profiles]
        df = pd.DataFrame(rows)
        labels, proba = predict_candidate(pipe, df)
        for (pid, d), lab, (_, pr) in zip(profiles, labels, proba.iterrows()):
            out(f"  {pid:<4}{d['age']:>4}{d['sex']:>4}{d['bmi']:>5}"
                f"{d['race_ethnicity']:>5}{d['family_history']:>4}"
                f"{d['hypertension']:>4}{d['physical_activity']:>4}"
                f"{d['smoking_status']:>4}  {lab:>8}"
                f"{pr['Low']:>9.3f}{pr['Moderate']:>9.3f}{pr['High']:>9.3f}")
        return labels, proba

    lab_low, prob_low = run_profiles("A. CLEAR LOW-RISK PROFILES", low_profiles)
    lab_mod, prob_mod = run_profiles("B. MODERATE/INTERMEDIATE PROFILES", mod_profiles)
    lab_high, prob_high = run_profiles("C. CLEAR HIGH-RISK PROFILES", high_profiles)
    lab_combo, prob_combo = run_profiles("D. COMBINATION PROFILES", combo_profiles)

    # ---------------- PART 3: stress grid ----------------
    out("\n" + "=" * 70)
    out("PART 3 - STRESS TEST GRID")
    out("=" * 70)

    ages = [20, 30, 40, 50, 60, 70, 80]
    bmis = [18, 22, 25, 28, 30, 35, 40, 45]
    fhs = [0, 1]
    htns = [0, 1]

    grid_rows = []
    for a in ages:
        for b in bmis:
            for fh in fhs:
                for htn in htns:
                    grid_rows.append(base(a, b, fh, htn))
    grid = pd.DataFrame([profile_row(d) for d in grid_rows])
    g_labels, g_proba = predict_candidate(pipe, grid)

    from collections import Counter
    cnt = Counter(g_labels)
    out(f"  Grid size: {len(grid_rows):,} (age x BMI x family_history x hypertension)")
    for cls in ["Low", "Moderate", "High"]:
        n = cnt.get(cls, 0)
        out(f"    {cls}: {n:,} ({n / len(grid_rows):.1%})")

    high_sub = ((grid["age"] >= 60) & (grid["bmi"] >= 35) &
                (grid["family_history"] == 1) & (grid["hypertension"] == 1))
    hs_labels = [g_labels[i] for i in np.where(high_sub.values)[0]]
    hs_p = g_proba.loc[high_sub.values, "High"]
    out(f"\n  HIGH-RISK SUBSET (age>=60 & bmi>=35 & fh=1 & htn=1):")
    out(f"    profiles: {int(high_sub.sum())}")
    hc = Counter(hs_labels)
    out(f"    predicted Low: {hc.get('Low', 0)}  Moderate: {hc.get('Moderate', 0)}  "
        f"High: {hc.get('High', 0)}")
    out(f"    P(High): min={hs_p.min():.3f}  max={hs_p.max():.3f}  mean={hs_p.mean():.3f}")

    low_sub = ((grid["age"] <= 30) & (grid["bmi"] <= 25) &
               (grid["family_history"] == 0) & (grid["hypertension"] == 0))
    ls_labels = [g_labels[i] for i in np.where(low_sub.values)[0]]
    ls_p = g_proba.loc[low_sub.values, "High"]
    out(f"\n  LOW-RISK SUBSET (age<=30 & bmi<=25 & fh=0 & htn=0):")
    out(f"    profiles: {int(low_sub.sum())}")
    lc = Counter(ls_labels)
    out(f"    predicted Low: {lc.get('Low', 0)}  Moderate: {lc.get('Moderate', 0)}  "
        f"High: {lc.get('High', 0)}")
    out(f"    P(High): min={ls_p.min():.3f}  max={ls_p.max():.3f}  mean={ls_p.mean():.3f}")

    # ---------------- PART 4: trend checks ----------------
    out("\n" + "=" * 70)
    out("PART 4 - TREND CHECKS (observed direction, no monotonic constraint)")
    out("=" * 70)

    fixed = base(60, 30, 1, 1)
    out("  Age trend (bmi=30, fh=1, htn=1, male, race1, pa=1, smoke=0):")
    age_means = []
    for a in [20, 30, 40, 50, 60, 70, 80]:
        row = dict(fixed); row["age"] = a
        _, pr = predict_candidate(pipe, pd.DataFrame([profile_row(row)]))
        age_means.append(pr["High"].iloc[0])
        out(f"    age {a:>3}: P(High)={age_means[-1]:.3f}")
    out(f"    -> generally increasing: "
        f"{age_means[-1] > age_means[0]} (start={age_means[0]:.3f}, end={age_means[-1]:.3f})")

    out("  BMI trend (age=60, fh=1, htn=1, male, race1, pa=1, smoke=0):")
    bmi_means = []
    for b in [18, 22, 25, 28, 30, 35, 40, 45]:
        row = dict(fixed); row["bmi"] = b
        _, pr = predict_candidate(pipe, pd.DataFrame([profile_row(row)]))
        bmi_means.append(pr["High"].iloc[0])
        out(f"    bmi {b:>3}: P(High)={bmi_means[-1]:.3f}")
    out(f"    -> generally increasing: "
        f"{bmi_means[-1] > bmi_means[0]} (start={bmi_means[0]:.3f}, end={bmi_means[-1]:.3f})")

    out("  Family-history trend (mean P(High) over the full grid):")
    fh1 = g_proba.loc[grid["family_history"] == 1, "High"].mean()
    fh0 = g_proba.loc[grid["family_history"] == 0, "High"].mean()
    out(f"    fh=1: {fh1:.3f}   fh=0: {fh0:.3f}   -> higher risk with FH: {fh1 > fh0}")

    out("  Hypertension trend (mean P(High) over the full grid):")
    ht1 = g_proba.loc[grid["hypertension"] == 1, "High"].mean()
    ht0 = g_proba.loc[grid["hypertension"] == 0, "High"].mean()
    out(f"    htn=1: {ht1:.3f}   htn=0: {ht0:.3f}   -> higher risk with HTN: {ht1 > ht0}")

    # ---------------- PART 5: old vs new comparison ----------------
    out("\n" + "=" * 70)
    out("PART 5 - OLD MODEL (Model 1B) vs NEW CANDIDATE")
    out("=" * 70)

    old = joblib.load(OLD_MODEL)
    # Old model step names are not assumed; penultimate step = preprocessor,
    # final step = classifier.
    prep_step = old.steps[-2][0]
    clf_step = old.steps[-1][0]
    old_cols = list(old.named_steps[prep_step].feature_names_in_)
    out(f"  Old model pipeline: {type(old).__name__} -> "
        f"{type(old.named_steps[clf_step]).__name__} | features: {old_cols}")

    compare_profiles = low_profiles + mod_profiles + high_profiles + [
        ("P1 high-risk", base(65, 40, 1, 1, race=4)),
        ("P2 high-risk", base(75, 45, 1, 1, sex=2, race=4)),
    ]

    out(f"\n  {'id':<26}{'new pred':>10}{'old pred':>10}{'new P(High)':>13}"
        f"{'old P(High)':>13}")
    diffs = []
    for pid, d in compare_profiles:
        # New candidate: full 8-predictor profile
        _, pr_new = predict_candidate(pipe, pd.DataFrame([profile_row(d)]))
        new_lab = pr_new["High"].iloc[0]

        # Old model: only its 5 frozen features; family_history 1.0 or NaN
        old_row = {
            "age": d["age"], "bmi": d["bmi"], "sex": d["sex"],
            "race_ethnicity": d["race_ethnicity"],
            "family_history": 1.0 if d["family_history"] == 1 else float("nan"),
        }
        old_df = pd.DataFrame([old_row])[old_cols]
        old_lab = old.predict(old_df)[0]            # old model returns class strings
        old_proba = old.predict_proba(old_df)[0]
        old_classes = list(old.named_steps[clf_step].classes_)
        old_phigh = old_proba[old_classes.index("High")]

        new_class = enc_to_label(pipe.predict(pd.DataFrame([profile_row(d)]))[0])
        old_phigh_f = float(old_phigh)
        new_phigh_f = float(new_lab)
        out(f"  {pid:<26}{new_class:>10}{old_lab:>10}{new_phigh_f:>13.4f}"
            f"{old_phigh_f:>13.4f}")
        diffs.append((pid, new_class, old_lab, new_phigh_f, old_phigh_f))

    out("\n  Focus on the two previously problematic profiles (high-risk "
        "combinations):")
    for pid, nc, oc, np_, op_ in diffs:
        if pid.startswith("P"):
            if np_ > op_ + 0.1:
                signal = "much stronger"
            elif np_ > op_:
                signal = "stronger"
            else:
                signal = "not stronger"
            out(f"    {pid}: new={nc} P(High)={np_:.4f} | old={oc} P(High)={op_:.4f} "
                f"-> candidate risk signal {signal}")

    # ---------------- PART 6: decision ----------------
    out("\n" + "=" * 70)
    out("PART 6 - VALIDATION DECISION")
    out("=" * 70)

    grid_all_three = all(cnt.get(c, 0) > 0 for c in ["Low", "Moderate", "High"])
    high_not_all_low = hc.get("High", 0) > 0
    separation = hs_p.mean() - ls_p.mean()
    trends_ok = (age_means[-1] > age_means[0] and bmi_means[-1] > bmi_means[0]
                 and fh1 > fh0 and ht1 > ht0)
    interface_ok = (fni == PREDICTORS and not overlap)

    out(f"  All three classes produced across grid: {grid_all_three}")
    out(f"  High-risk subset not universally Low: {high_not_all_low}")
    out(f"  Mean P(High) separation high-risk vs low-risk: {separation:+.3f}")
    out(f"  Sensible trend directions: {trends_ok}")
    out(f"  Encoding/pipeline clean + exact 8-predictor interface: {interface_ok}")

    passed = (grid_all_three and high_not_all_low and separation > 0.2
              and trends_ok and interface_ok)
    out(f"\n  FINAL DECISION: {'PASS' if passed else 'FAIL'}")
    if not passed:
        reasons = []
        if not grid_all_three: reasons.append("grid lacks a class")
        if not high_not_all_low: reasons.append("high-risk all Low")
        if separation <= 0.2: reasons.append("weak P(High) separation")
        if not trends_ok: reasons.append("trend direction issue")
        if not interface_ok: reasons.append("interface issue")
        out(f"  Reason: {', '.join(reasons)}")

    REPORT_OUT.write_text("\n".join(LINES), encoding="utf-8")
    out(f"\nValidation report saved: {REPORT_OUT}")


if __name__ == "__main__":
    main()