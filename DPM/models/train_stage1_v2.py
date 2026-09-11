"""
DiaBeta V2 - Stage 1 Screener Training (v2, 8 predictors)
==========================================================

Trains and compares multiclass (Low / Moderate / High) classifiers for the
Stage 1 NON-LABORATORY diabetes screener using the audited dataset:

    DPM/data/stage1_v2_nhanes_8predictors.csv

The model learns ONLY from the 8 locked predictors. diq010 is target
provenance only and is never a feature. No laboratory variable is used.
Imputation/encoding are fitted ONLY on training folds (no leakage).

Models compared (all with reproducible seeds):
    - RandomForestClassifier
    - XGBClassifier (multiclass)
    - HistGradientBoostingClassifier
    - RandomForest + SMOTE (imblearn Pipeline; SMOTE strictly inside
      cross-validation training folds, never on validation/test)

Model selection: mean 5-fold stratified CV Macro F1 on the training split.
The held-out test split is untouched until final evaluation of every
candidate. No probability/threshold manipulation.

Outputs (NEW files; the old model is NOT overwritten):
    DPM/models/stage1_v2_model.pkl            (selected pipeline, joblib)
    DPM/models/stage1_v2_training_report.txt  (full experiment report)
    DPM/models/stage1_v2_test_predictions.csv (test y_true/y_pred/proba)
"""

from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent   # .../DiaBeta/DPM
DATA_FILE = BASE_DIR / "data" / "stage1_v2_nhanes_8predictors.csv"

MODEL_OUT = BASE_DIR / "models" / "stage1_v2_model.pkl"
REPORT_OUT = BASE_DIR / "models" / "stage1_v2_training_report.txt"
PRED_OUT = BASE_DIR / "models" / "stage1_v2_test_predictions.csv"

PREDICTORS = [
    "age", "sex", "bmi", "race_ethnicity",
    "family_history", "hypertension", "physical_activity", "smoking_status",
]

NUMERIC_FEATURES = ["age", "bmi"]
CATEGORICAL_FEATURES = [
    "sex", "race_ethnicity", "family_history",
    "hypertension", "physical_activity", "smoking_status",
]

# Never allowed in X (target provenance / identifiers / laboratory).
FORBIDDEN = {"diq010", "SEQN", "cycle", "hba1c", "lbxgh",
             "glucose", "fasting_glucose", "lbxglu"}

TARGET = "diabetes_risk"
CLASSES = ["Low", "Moderate", "High"]

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

TRAIN_REPORT_LINES = []


def out(text=""):
    print(text)
    TRAIN_REPORT_LINES.append(text)


# ============================================================
# DATA LOADING + LEAKAGE GUARDS
# ============================================================

def load_data():
    df = pd.read_csv(DATA_FILE)

    # Leakage guard 1: X must be EXACTLY the 8 locked predictors.
    missing = [c for c in PREDICTORS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing predictor columns: {missing}")
    extra = [c for c in df.columns if c not in PREDICTORS + [TARGET, "diq010", "SEQN", "cycle"]]
    if extra:
        raise ValueError(f"Unexpected columns in dataset: {extra}")

    X = df[PREDICTORS].copy()
    y = df[TARGET].copy()

    # Leakage guard 2: forbidden columns must not be in X.
    overlap = set(X.columns) & FORBIDDEN
    if overlap:
        raise ValueError(f"Forbidden column(s) leaked into X: {sorted(overlap)}")

    # Leakage guard 3: target must have no unknown classes.
    unknown = set(y.unique()) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unexpected target values: {unknown}")
    if y.isna().any():
        raise ValueError("Target contains missing values.")

    return X, y


# ============================================================
# PREPROCESSING (fitted only on training folds)
# ============================================================

def make_preprocessor():
    """Numeric: median impute + standardize. Categorical: most-frequent
    impute + one-hot (unknown categories tolerated at inference)."""

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def make_candidates():
    """Candidate (name, pipeline). SMOTE, when used, lives INSIDE the
    pipeline so it only ever sees training folds."""

    candidates = []

    rf = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    )
    candidates.append((
        "RandomForest",
        Pipeline([("prep", make_preprocessor()), ("clf", rf)]),
    ))

    if XGB_AVAILABLE:
        xgb = XGBClassifier(
            objective="multi:softprob",
            n_estimators=300,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            n_jobs=-1,
            tree_method="hist",
        )
        candidates.append((
            "XGBoost",
            Pipeline([("prep", make_preprocessor()), ("clf", xgb)]),
        ))

    hgb = HistGradientBoostingClassifier(
        max_iter=300, random_state=RANDOM_STATE
    )
    candidates.append((
        "HistGradientBoosting",
        Pipeline([("prep", make_preprocessor()), ("clf", hgb)]),
    ))

    if IMBLEARN_AVAILABLE:
        rf_smote = RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        )
        candidates.append((
            "RandomForest+SMOTE",
            ImbPipeline([
                ("prep", make_preprocessor()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("clf", rf_smote),
            ]),
        ))

    return candidates


# ============================================================
# CV MODEL SELECTION
# ============================================================

def run_cv(candidates, X_train, y_train_enc):
    """5-fold stratified CV; selection metric = mean Macro F1."""

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    out("\n" + "=" * 70)
    out("CROSS-VALIDATION (model selection)")
    out("=" * 70)

    results = {}
    for name, pipe in candidates:
        t0 = time.time()
        scores = cross_validate(
            pipe, X_train, y_train_enc, cv=cv,
            scoring={
                "f1_macro": "f1_macro",
                "recall_macro": "recall_macro",
                "accuracy": "accuracy",
                "roc_auc_ovr": "roc_auc_ovr",
            },
            n_jobs=1,
        )
        elapsed = time.time() - t0
        results[name] = {
            "f1_macro": scores["test_f1_macro"].mean(),
            "f1_macro_std": scores["test_f1_macro"].std(),
            "recall_macro": scores["test_recall_macro"].mean(),
            "accuracy": scores["test_accuracy"].mean(),
            "roc_auc_ovr": scores["test_roc_auc_ovr"].mean(),
            "elapsed_s": elapsed,
        }
        r = results[name]
        out(f"  {name:<24} CV macro-F1={r['f1_macro']:.4f} "
            f"(+/-{r['f1_macro_std']:.4f})  recall={r['recall_macro']:.4f}  "
            f"acc={r['accuracy']:.4f}  ROC-AUC(ovr)={r['roc_auc_ovr']:.4f}  "
            f"({elapsed:.0f}s)")

    best_name = max(results, key=lambda n: results[n]["f1_macro"])
    out(f"\n  SELECTED by CV macro-F1: {best_name}")
    return best_name, results


# ============================================================
# FINAL TEST EVALUATION (test set untouched until here)
# ============================================================

def evaluate_on_test(name, pipe, X_train, y_train_enc, X_test, y_test, le):
    """Fit on encoded labels, predict, map back to class strings."""

    pipe.fit(X_train, y_train_enc)
    y_pred_enc = pipe.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    proba = None
    try:
        proba = pipe.predict_proba(X_test)
    except Exception:
        proba = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "y_pred": y_pred,
        "proba": proba,
        "classes": list(le.classes_),
    }

    if proba is not None:
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_test, proba, multi_class="ovr", average="macro",
                labels=sorted(CLASSES),  # sklearn requires ordered labels
            )
        except Exception:
            metrics["roc_auc_ovr"] = np.nan

    # Per-class metrics
    report = classification_report(
        y_test, y_pred, labels=CLASSES, zero_division=0, output_dict=True
    )
    metrics["per_class"] = {
        cls: {
            "precision": report[cls]["precision"],
            "recall": report[cls]["recall"],
            "f1": report[cls]["f1-score"],
            "support": report[cls]["support"],
        }
        for cls in CLASSES
    }
    metrics["confusion"] = confusion_matrix(y_test, y_pred, labels=CLASSES)

    return metrics


def report_test_results(name, m):
    out(f"\n  {name}")
    out("-" * 70)
    out(f"  Accuracy:          {m['accuracy']:.4f}")
    out(f"  Macro Precision:   {m['precision_macro']:.4f}")
    out(f"  Macro Recall:      {m['recall_macro']:.4f}")
    out(f"  Macro F1:          {m['f1_macro']:.4f}")
    if "roc_auc_ovr" in m and not np.isnan(m["roc_auc_ovr"]):
        out(f"  ROC-AUC (macro ovr): {m['roc_auc_ovr']:.4f}")
    out("  Per-class (Precision / Recall / F1):")
    for cls in CLASSES:
        pc = m["per_class"][cls]
        out(f"    {cls:<9} {pc['precision']:.4f} / {pc['recall']:.4f} / "
            f"{pc['f1']:.4f}   (n={int(pc['support'])})")
    cm = m["confusion"]
    out("  Confusion matrix (rows=true, cols=pred, order Low/Moderate/High):")
    for i, row in enumerate(cm):
        out(f"    {CLASSES[i]:<9} " + " ".join(f"{v:6d}" for v in row))


# ============================================================
# MAIN
# ============================================================

def main():
    t_start = time.time()

    out("=" * 70)
    out("DIA BETA V2 - STAGE 1 SCREENER TRAINING (v2, 8 predictors)")
    out("=" * 70)

    # ---- 1. Load + guards ----
    X, y = load_data()

    out(f"\nDataset: {DATA_FILE.name}")
    out(f"Rows: {len(X):,}")
    out(f"Predictors ({len(PREDICTORS)}): {', '.join(PREDICTORS)}")
    out(f"Target: {TARGET}  ({', '.join(CLASSES)})")
    out("Leakage guards: diq010/SEQN/cycle/labs excluded from X - VERIFIED")

    counts = y.value_counts().reindex(CLASSES)
    for cls in CLASSES:
        n = int(counts[cls])
        out(f"  {cls}: {n:,} ({n / len(y):.2%})")

    # ---- 2. Stratified split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    out(f"\nTrain/test split: {len(X_train):,} / {len(X_test):,} "
        f"(stratified, random_state={RANDOM_STATE})")

    # ---- 3. Missing values ----
    out("\nMissing-value handling (fitted on training folds only):")
    out(f"  Numeric ({', '.join(NUMERIC_FEATURES)}): median imputation + standardization")
    out(f"  Categorical ({', '.join(CATEGORICAL_FEATURES)}): most-frequent imputation + one-hot encoding")
    out("  Missing predictor values are NOT dropped; imputers are learned per fold.")

    # ---- 4. Candidates ----
    candidates = make_candidates()
    out(f"\nCandidates evaluated: {', '.join(n for n, _ in candidates)}")
    if IMBLEARN_AVAILABLE:
        out("SMOTE: tested as one candidate inside the CV pipeline only "
            "(RandomForest+SMOTE); never applied to validation/test.")
    else:
        out("SMOTE: not available in this environment; class imbalance "
            "addressed via documented selection emphasis.")

    # ---- 5. Encode target (uniform for all candidates incl. XGBoost) ----
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    # ---- 6. CV selection ----
    best_name, cv_results = run_cv(candidates, X_train, y_train_enc)

    # ---- 7. Final test evaluation for EVERY candidate ----
    out("\n" + "=" * 70)
    out("FINAL HELD-OUT TEST EVALUATION (per candidate)")
    out("=" * 70)

    test_results = {}
    for name, pipe in candidates:
        m = evaluate_on_test(name, pipe, X_train, y_train_enc, X_test, y_test, le)
        test_results[name] = m
        report_test_results(name, m)

    # ---- 8. Selection summary ----
    out("\n" + "=" * 70)
    out("MODEL SELECTION SUMMARY")
    out("=" * 70)
    out(f"{'Candidate':<24}{'CV F1':>10}{'Test F1':>10}{'Test rec':>10}"
        f"{'Mod F1':>9}{'High F1':>9}")
    for name, _ in candidates:
        cv = cv_results[name]
        m = test_results[name]
        out(f"{name:<24}{cv['f1_macro']:>10.4f}{m['f1_macro']:>10.4f}"
            f"{m['recall_macro']:>10.4f}"
            f"{m['per_class']['Moderate']['f1']:>9.4f}"
            f"{m['per_class']['High']['f1']:>9.4f}")

    out(f"\nBEST MODEL (CV macro-F1): {best_name}")

    # ---- 9. Save artifacts ----
    best_pipe = dict(candidates)[best_name]
    best_pipe.fit(X_train, y_train_enc)
    joblib.dump(best_pipe, MODEL_OUT)
    out(f"\nSaved pipeline: {MODEL_OUT}")

    # ---- 10. Save test predictions/probabilities ----
    m_best = test_results[best_name]
    pred_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": m_best["y_pred"],
    })
    if m_best["proba"] is not None:
        for i, cls in enumerate(m_best["classes"]):
            pred_df[f"p_{cls}"] = m_best["proba"][:, i]
    pred_df = pred_df.reset_index(drop=True)
    pred_df.to_csv(PRED_OUT, index=False)
    out(f"Saved test predictions: {PRED_OUT}")

    # ---- 11. Final report block ----
    m_final = test_results[best_name]
    out("\n" + "=" * 70)
    out("FINAL SELECTED MODEL - HELD-OUT TEST PERFORMANCE")
    out("=" * 70)
    out(f"Model: {best_name}")
    report_test_results(best_name, m_final)

    out("\nOBSERVATIONS / LIMITATIONS")
    out("-" * 70)
    out("1. Class imbalance is mild (Low:Moderate:High = 47.5%:33.7%:18.8%,"
        " max ratio ~2.5:1); SMOTE was evaluated as a candidate inside CV only.")
    out("2. The pipeline is a single loadable artifact: preprocessing and")
    out("   classifier are identical at training and inference time.")
    out("3. This model is NOT yet integrated into the application; the frozen")
    out("   DPM/models/diabeta_dataset1_model.pkl remains untouched.")
    out(f"\nTotal elapsed: {time.time() - t_start:.0f}s")

    # ---- 12. Write report file ----
    REPORT_OUT.write_text("\n".join(TRAIN_REPORT_LINES), encoding="utf-8")
    out(f"\nReport saved: {REPORT_OUT}")

    out("\n" + "=" * 70)
    out("STAGE 1 V2 TRAINING COMPLETE")
    out("=" * 70)


if __name__ == "__main__":
    main()