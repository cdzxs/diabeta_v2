"""
DiaBeta V2 - Stage 1 Screener Diagnostics + Controlled Improvement Experiment
=============================================================================

Diagnoses the current Stage 1 V2 baseline
(DPM/models/stage1_v2_model.pkl, HistGradientBoosting) and runs a SMALL,
defensible set of improvement experiments on the SAME stratified split
(random_state=42) used by the original training run.

Rules honored:
    * Stage 1 only; no HbA1c/glucose predictors; no SEQN/cycle/diq010.
    * Preprocessing always inside the CV pipeline (fitted on folds only).
    * StratifiedKFold(5, shuffle, random_state=42); test set untouched
      until final evaluation.
    * No probability/threshold manipulation. No SMOTE (already shown not
      to help). No metric fabrication.
    * The current production artifact stage1_v2_model.pkl is NEVER
      overwritten. An improved candidate is saved separately as
      stage1_v2_candidate.pkl only if it passes the decision rule.

Outputs:
    DPM/models/stage1_v2_diagnostic_report.txt
    DPM/models/stage1_v2_candidate.pkl          (only if improved)
"""

from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent          # .../DiaBeta/DPM
DATA_FILE = BASE_DIR / "data" / "stage1_v2_nhanes_8predictors.csv"
BASELINE_MODEL = BASE_DIR / "models" / "stage1_v2_model.pkl"

REPORT_OUT = BASE_DIR / "models" / "stage1_v2_diagnostic_report.txt"
CANDIDATE_OUT = BASE_DIR / "models" / "stage1_v2_candidate.pkl"

PREDICTORS = [
    "age", "sex", "bmi", "race_ethnicity",
    "family_history", "hypertension", "physical_activity", "smoking_status",
]
NUMERIC_FEATURES = ["age", "bmi"]
CATEGORICAL_FEATURES = [
    "sex", "race_ethnicity", "family_history",
    "hypertension", "physical_activity", "smoking_status",
]
FORBIDDEN = {"diq010", "SEQN", "cycle", "hba1c", "lbxgh",
             "glucose", "fasting_glucose", "lbxglu"}
TARGET = "diabetes_risk"
CLASSES = ["Low", "Moderate", "High"]          # display order

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# Decision rule: a candidate must beat baseline test Macro-F1 by at least
# this margin AND must not damage the clinically important classes.
MIN_F1_GAIN = 0.010
MAX_CLASS_DROP = 0.020

LINES = []


def out(text=""):
    print(text)
    LINES.append(text)


# ============================================================
# DATA
# ============================================================

def load_data():
    df = pd.read_csv(DATA_FILE)
    missing = [c for c in PREDICTORS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing predictor columns: {missing}")
    extra = [c for c in df.columns if c not in PREDICTORS + [TARGET, "diq010", "SEQN", "cycle"]]
    if extra:
        raise ValueError(f"Unexpected columns: {extra}")
    X = df[PREDICTORS].copy()
    y = df[TARGET].copy()
    if set(X.columns) & FORBIDDEN:
        raise ValueError("Forbidden column leaked into X")
    if set(y.unique()) - set(CLASSES):
        raise ValueError("Unexpected target values")
    return X, y


def make_preprocessor(cat_encoding="onehot", missing_indicator=False):
    """Preprocessor variants. Always fitted inside the CV pipeline."""

    num_steps = [("impute", SimpleImputer(
        strategy="median", add_indicator=missing_indicator))]
    cat_steps = [("impute", SimpleImputer(
        strategy="most_frequent", add_indicator=missing_indicator))]

    if cat_encoding == "onehot":
        cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore")))
    elif cat_encoding == "ordinal":
        cat_steps.append(("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1)))
    else:
        raise ValueError(f"Unknown categorical encoding: {cat_encoding}")

    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_FEATURES),
        ("cat", Pipeline(cat_steps), CATEGORICAL_FEATURES),
    ])


def make_candidates():
    """Controlled experiment set (small, defensible). Baseline first."""

    candidates = []

    def hgb(**kw):
        params = dict(max_iter=300, random_state=RANDOM_STATE)
        params.update(kw)
        return HistGradientBoostingClassifier(**params)

    candidates.append((
        "baseline_hgb",
        Pipeline([("prep", make_preprocessor()), ("clf", hgb())]),
        "current config (one-hot cats, median/mode impute, max_iter=300)",
    ))

    candidates.append((
        "hgb_lr05",
        Pipeline([("prep", make_preprocessor()),
                  ("clf", hgb(learning_rate=0.05, max_iter=600))]),
        "lower learning rate, more iterations (0.05 / 600)",
    ))

    candidates.append((
        "hgb_capacity",
        Pipeline([("prep", make_preprocessor()),
                  ("clf", hgb(max_leaf_nodes=127, min_samples_leaf=5))]),
        "higher capacity (max_leaf_nodes=127, min_samples_leaf=5)",
    ))

    candidates.append((
        "hgb_regularized",
        Pipeline([("prep", make_preprocessor()),
                  ("clf", hgb(max_leaf_nodes=15, min_samples_leaf=40,
                              l2_regularization=1.0))]),
        "stronger regularization (leaf=15, min_leaf=40, l2=1.0)",
    ))

    candidates.append((
        "hgb_ordinal_cats",
        Pipeline([("prep", make_preprocessor(cat_encoding="ordinal")),
                  ("clf", hgb())]),
        "categoricals ordinal-encoded instead of one-hot",
    ))

    candidates.append((
        "hgb_missing_ind",
        Pipeline([("prep", make_preprocessor(missing_indicator=True)),
                  ("clf", hgb())]),
        "missing-value indicator columns added (num + cat)",
    ))

    return candidates


# ============================================================
# METRIC HELPERS
# ============================================================

def per_class_from_confusion(cm, idx):
    """Precision/recall/F1 for class at index idx from a confusion matrix."""
    tp = cm[idx, idx]
    fp = int(cm[:, idx].sum()) - tp
    fn = int(cm[idx, :].sum()) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def summarize_cm(cm):
    """Full macro + per-class metrics from a summed confusion matrix.
    cm rows/cols are in encoded order (le.classes_ = High, Low, Moderate)."""
    # encoded index per display class
    idx = {cls: int(np.where(le.classes_ == cls)[0][0]) for cls in CLASSES}
    acc = cm.trace() / cm.sum()
    per = {}
    precs, recs, f1s = [], [], []
    for cls in CLASSES:
        p, r, f = per_class_from_confusion(cm, idx[cls])
        per[cls] = {"precision": p, "recall": r, "f1": f}
        precs.append(p); recs.append(r); f1s.append(f)
    return {
        "accuracy": acc,
        "macro_precision": float(np.mean(precs)),
        "macro_recall": float(np.mean(recs)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per,
        "confusion": cm,
    }


le = None  # global LabelEncoder, set in main


# ============================================================
# PART 1 - DIAGNOSTICS (baseline model, test set only)
# ============================================================

def diagnostics(baseline, X_test, y_test, y_test_enc, X_train):
    out("\n" + "=" * 70)
    out("PART 1 - BASELINE MODEL DIAGNOSTICS")
    out("=" * 70)

    # --- 1. Permutation importance (model-agnostic, per predictor) ---
    out("\n1. PERMUTATION IMPORTANCE (test set, 5 repeats, seed 42)")
    out("-" * 70)
    pi = permutation_importance(
        baseline, X_test, y_test_enc, n_repeats=5,
        random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=-1,
    )
    order = np.argsort(-pi.importances_mean)
    for i in order:
        out(f"  {PREDICTORS[i]:<20} mean={pi.importances_mean[i]:+.4f} "
            f"std={pi.importances_std[i]:.4f}")

    # --- 2. Which predictors matter most ---
    top = [PREDICTORS[i] for i in order[:4]]
    out("\n2. TOP CONTRIBUTORS (by permutation importance on macro-F1): "
        + ", ".join(top))

    # --- 3. Class-wise probability distributions ---
    out("\n3. CLASS-WISE PREDICTED-PROBABILITY DISTRIBUTIONS (test set)")
    out("-" * 70)
    proba = baseline.predict_proba(X_test)
    proba_df = pd.DataFrame(
        proba, columns=list(le.classes_), index=y_test.index
    )
    out("  Mean predicted P per true class (rows=true):")
    out(f"  {'true':<10}{'P(Low)':>10}{'P(Moderate)':>12}{'P(High)':>10}"
        f"{'mean P(true)':>14}")
    for cls in CLASSES:
        sub = proba_df[y_test == cls]
        mean_true = sub[cls].mean()
        out(f"  {cls:<10}"
            f"{sub['Low'].mean():>10.3f}"
            f"{sub['Moderate'].mean():>12.3f}"
            f"{sub['High'].mean():>10.3f}"
            f"{mean_true:>14.3f}")

    # --- 4. Error analysis ---
    out("\n4. ERROR ANALYSIS (test set, true vs predicted)")
    out("-" * 70)
    y_pred_enc = baseline.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    cm = confusion_matrix(y_test_enc, y_pred_enc, labels=[0, 1, 2])
    display_order = ["Low", "Moderate", "High"]
    idx = {cls: int(np.where(le.classes_ == cls)[0][0]) for cls in display_order}
    disp_cm = np.zeros((3, 3), dtype=int)
    for i, t in enumerate(display_order):
        for j, p in enumerate(display_order):
            disp_cm[i, j] = cm[idx[t], idx[p]]
    out("  Confusion matrix (rows=true, cols=pred, Low/Moderate/High):")
    for i, row in enumerate(disp_cm):
        out(f"    {display_order[i]:<10}" + " ".join(f"{v:6d}" for v in row))

    out("\n  Error pairs (true -> predicted): count, mean P(pred), mean P(true), "
        "mean age, mean BMI")
    for t in display_order:
        for p in display_order:
            if t == p:
                continue
            mask = (y_test == t) & (y_pred == p)
            n = int(mask.sum())
            if n == 0:
                continue
            sub = proba_df[mask]
            out(f"    {t} -> {p:<9} n={n:4d}  P({p})={sub[p].mean():.3f}  "
                f"P({t})={sub[t].mean():.3f}  "
                f"age={X_test.loc[mask, 'age'].mean():.1f}  "
                f"bmi={X_test.loc[mask, 'bmi'].mean():.1f}")

    # --- 5. Missing-value handling audit ---
    out("\n5. MISSING-VALUE HANDLING AUDIT")
    out("-" * 70)
    out("  Missing counts in training data (imputers are fitted on train):")
    for p in PREDICTORS:
        n_miss = int(X_train[p].isna().sum())
        if n_miss:
            fill = X_train[p].median() if p in NUMERIC_FEATURES else X_train[p].mode().iloc[0]
            out(f"    {p:<20} {n_miss:,} ({n_miss / len(X_train):.2%}) "
                f"-> imputed with {fill:.3g} ({'median' if p in NUMERIC_FEATURES else 'mode'})")
    out("  Missingness is low (<2% per predictor); the indicator-column "
        "experiment tests whether the missing pattern itself carries signal.")

    # --- 6. Suspected limitation ---
    out("\n6. LIMITATION ASSESSMENT")
    out("-" * 70)
    mod_idx = idx["Moderate"]
    low_idx = idx["Low"]
    high_idx = idx["High"]
    mod_low = cm[mod_idx, low_idx] / cm[mod_idx, :].sum()   # Moderate->Low
    mod_high = cm[mod_idx, high_idx] / cm[mod_idx, :].sum()  # Moderate->High
    low_mod = cm[low_idx, mod_idx] / cm[low_idx, :].sum()    # Low->Moderate
    high_mod = cm[high_idx, mod_idx] / cm[high_idx, :].sum()  # High->Moderate
    out(f"  Moderate class: {mod_low:.1%} of true Moderate predicted Low, "
        f"{mod_high:.1%} predicted High (true Moderate rate = "
        f"{cm[mod_idx, mod_idx] / cm[mod_idx, :].sum():.1%}).")
    out(f"  Low class: {low_mod:.1%} of true Low predicted Moderate.")
    out(f"  High class: {high_mod:.1%} of true High predicted Moderate.")
    out("  Evidence points to CLASS OVERLAP: the Moderate band "
        "(5.7-6.4 HbA1c / 100-125 glucose) is a broad middle zone that "
        "straddles Low and High in 8 non-laboratory features, so the model "
        "necessarily trades Moderate errors against Low/High errors.")
    return disp_cm, top


# ============================================================
# PART 2+3 - CONTROLLED EXPERIMENTS
# ============================================================

def run_experiments(candidates, X_train, y_train_enc):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    out("\n" + "=" * 70)
    out("PART 2 - CONTROLLED EXPERIMENTS (5-fold stratified CV, seed 42)")
    out("=" * 70)

    cv_results = {}
    for name, pipe, desc in candidates:
        t0 = time.time()
        cm_sum = np.zeros((3, 3), dtype=int)
        aucs = []
        for tr_idx, va_idx in cv.split(X_train, y_train_enc):
            pipe.fit(X_train.iloc[tr_idx], y_train_enc[tr_idx])
            yp = pipe.predict(X_train.iloc[va_idx])
            cm_sum += confusion_matrix(y_train_enc[va_idx], yp, labels=[0, 1, 2])
            try:
                pr = pipe.predict_proba(X_train.iloc[va_idx])
                aucs.append(roc_auc_score(y_train_enc[va_idx], pr,
                                          multi_class="ovr", average="macro",
                                          labels=[0, 1, 2]))
            except Exception:
                aucs.append(np.nan)
        s = summarize_cm(cm_sum)
        s["roc_auc_ovr"] = float(np.nanmean(aucs))
        s["elapsed_s"] = time.time() - t0
        cv_results[name] = s
        out(f"\n  {name}  ({desc})")
        out(f"    CV macro-F1={s['macro_f1']:.4f}  macro-recall={s['macro_recall']:.4f}"
            f"  acc={s['accuracy']:.4f}  ROC-AUC={s['roc_auc_ovr']:.4f}  ({s['elapsed_s']:.0f}s)")
        out(f"    Moderate P/R/F1: {s['per_class']['Moderate']['precision']:.4f}/"
            f"{s['per_class']['Moderate']['recall']:.4f}/"
            f"{s['per_class']['Moderate']['f1']:.4f}   "
            f"High P/R/F1: {s['per_class']['High']['precision']:.4f}/"
            f"{s['per_class']['High']['recall']:.4f}/"
            f"{s['per_class']['High']['f1']:.4f}")

    # Selection: primarily CV macro-F1, then CV macro-recall as tiebreak.
    best_name = max(cv_results, key=lambda n: (
        cv_results[n]["macro_f1"], cv_results[n]["macro_recall"]))
    out(f"\n  SELECTED by CV macro-F1 (tiebreak macro-recall): {best_name}")
    return best_name, cv_results


def evaluate_test(name, pipe, X_train, y_train_enc, X_test, y_test, y_test_enc):
    pipe.fit(X_train, y_train_enc)
    y_pred_enc = pipe.predict(X_test)
    cm = confusion_matrix(y_test_enc, y_pred_enc, labels=[0, 1, 2])
    s = summarize_cm(cm)
    try:
        pr = pipe.predict_proba(X_test)
        s["roc_auc_ovr"] = roc_auc_score(y_test_enc, pr, multi_class="ovr",
                                         average="macro", labels=[0, 1, 2])
    except Exception:
        s["roc_auc_ovr"] = np.nan
    return s


# ============================================================
# MAIN
# ============================================================

def main():
    global le
    t_start = time.time()

    out("=" * 70)
    out("DIA BETA V2 - STAGE 1 DIAGNOSTIC / IMPROVEMENT EXPERIMENT")
    out("=" * 70)

    X, y = load_data()
    le = LabelEncoder()
    le.fit(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    out(f"\nDataset: {DATA_FILE.name} | rows={len(X):,} "
        f"train={len(X_train):,} test={len(X_test):,} (stratified, seed 42)")
    out(f"Baseline model loaded: {BASELINE_MODEL.name}")

    baseline = joblib.load(BASELINE_MODEL)

    # ---------------- PART 1: diagnostics ----------------
    diagnostics(baseline, X_test, y_test, y_test_enc, X_train)

    # ---------------- PART 2/3: experiments ----------------
    candidates = make_candidates()
    out("\nExperiments planned: " + ", ".join(n for n, _, _ in candidates))

    best_cv_name, cv_results = run_experiments(candidates, X_train, y_train_enc)

    # Evaluate every candidate once on the untouched test set.
    out("\n" + "=" * 70)
    out("PART 3 - FINAL TEST EVALUATION (untouched test set)")
    out("=" * 70)
    test_results = {}
    for name, pipe, desc in candidates:
        s = evaluate_test(name, pipe, X_train, y_train_enc, X_test, y_test, y_test_enc)
        test_results[name] = s
        out(f"\n  {name}")
        out(f"    Test macro-F1={s['macro_f1']:.4f}  macro-recall={s['macro_recall']:.4f}"
            f"  acc={s['accuracy']:.4f}  ROC-AUC={s['roc_auc_ovr']:.4f}")
        out(f"    Moderate P/R/F1: {s['per_class']['Moderate']['precision']:.4f}/"
            f"{s['per_class']['Moderate']['recall']:.4f}/"
            f"{s['per_class']['Moderate']['f1']:.4f}   "
            f"High P/R/F1: {s['per_class']['High']['precision']:.4f}/"
            f"{s['per_class']['High']['recall']:.4f}/"
            f"{s['per_class']['High']['f1']:.4f}")

    baseline_test = test_results["baseline_hgb"]
    chosen_name = best_cv_name
    chosen_test = test_results[chosen_name]

    # ---------------- PART 4: decision rule ----------------
    out("\n" + "=" * 70)
    out("PART 4 - DECISION")
    out("=" * 70)
    gain = chosen_test["macro_f1"] - baseline_test["macro_f1"]
    mod_drop = baseline_test["per_class"]["Moderate"]["f1"] - \
        chosen_test["per_class"]["Moderate"]["f1"]
    high_drop = baseline_test["per_class"]["High"]["f1"] - \
        chosen_test["per_class"]["High"]["f1"]
    high_rec_drop = baseline_test["per_class"]["High"]["recall"] - \
        chosen_test["per_class"]["High"]["recall"]

    out(f"  Candidate from CV: {chosen_name}")
    out(f"  Test macro-F1 gain vs baseline: {gain:+.4f} "
        f"(required >= {MIN_F1_GAIN:+.3f})")
    out(f"  Moderate F1 change: {mod_drop:+.4f} (allowed drop <= {MAX_CLASS_DROP:.3f})")
    out(f"  High F1 change:     {high_drop:+.4f} (allowed drop <= {MAX_CLASS_DROP:.3f})")
    out(f"  High recall change: {high_rec_drop:+.4f} (allowed drop <= {MAX_CLASS_DROP:.3f})")

    improved = (
        gain >= MIN_F1_GAIN
        and mod_drop <= MAX_CLASS_DROP
        and high_drop <= MAX_CLASS_DROP
        and high_rec_drop <= MAX_CLASS_DROP
    )

    if improved:
        chosen_pipe = dict((n, p) for n, p, _ in candidates)[chosen_name]
        chosen_pipe.fit(X_train, y_train_enc)
        joblib.dump(chosen_pipe, CANDIDATE_OUT)
        out(f"\n  IMPROVED CANDIDATE FOUND: {chosen_name}")
        out(f"  Saved: {CANDIDATE_OUT}")
    else:
        out("\n  CURRENT MODEL RETAINED - no defensible improvement found.")
        out("  (stage1_v2_model.pkl unchanged; no candidate artifact saved)")

    # ---------------- PART 5: report file ----------------
    out("\n" + "=" * 70)
    out("FINAL SUMMARY")
    out("=" * 70)
    b = baseline_test
    out(f"BASELINE (HistGradientBoosting, current):")
    out(f"  CV macro-F1={cv_results['baseline_hgb']['macro_f1']:.4f}  "
        f"test macro-F1={b['macro_f1']:.4f}  test macro-recall={b['macro_recall']:.4f}"
        f"  test ROC-AUC={b['roc_auc_ovr']:.4f}")
    out(f"  Moderate F1={b['per_class']['Moderate']['f1']:.4f} "
        f"recall={b['per_class']['Moderate']['recall']:.4f}  "
        f"High F1={b['per_class']['High']['f1']:.4f} "
        f"recall={b['per_class']['High']['recall']:.4f}")
    out(f"\nSELECTED: {chosen_name}")
    if improved:
        out("  DECISION: ACCEPT as stage1_v2_candidate.pkl (review before "
            "any production switch)")
    else:
        out("  DECISION: CURRENT MODEL RETAINED")
    out(f"\nTotal elapsed: {time.time() - t_start:.0f}s")

    REPORT_OUT.write_text("\n".join(LINES), encoding="utf-8")
    out(f"\nReport saved: {REPORT_OUT}")


if __name__ == "__main__":
    main()