"""
hybrid_assessment.py

DiaBeta 2.0 - Assessment Layer (Stage 2)

Combines the VALIDATED Stage 1 V2 screening pipeline
(DPM/models/stage1_v2_model.pkl - an 8-predictor, non-laboratory
HistGradientBoosting pipeline) with the verified original hybrid rule set that
incorporates HbA1c and fasting glucose. This module contains NO machine learning
of its own - it is deterministic Python logic that calls the Stage 1 V2 pipeline
and applies clinical-threshold rules on top of its output.

The Stage 1 V2 pipeline is loaded read-only from disk and is never fitted,
refit, or altered by this module.
"""

import math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import joblib
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
# Active Stage 2 ML component: the validated Stage 1 V2 screener (also used by
# Stage 1). The retired Model 1B artifact (diabeta_dataset1_model.pkl) is no
# longer loaded anywhere in this module.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'stage1_v2_model.pkl'

# The eight, and only eight, predictors the Stage 1 V2 pipeline accepts, in the
# exact order the pipeline was trained on.
STAGE1_V2_FEATURES = [
    "age", "sex", "bmi", "race_ethnicity", "family_history",
    "hypertension", "physical_activity", "smoking_status",
]

# Stage 1 V2 was trained on LabelEncoder-encoded targets. sklearn's LabelEncoder
# sorts class names alphabetically, so encoded class 0 = "High", 1 = "Low",
# 2 = "Moderate". The saved pipeline's classifier.classes_ are therefore
# [0, 1, 2] and predict_proba() columns follow that exact order.
STAGE1_V2_CLASS_LABELS = {0: 'High', 1: 'Low', 2: 'Moderate'}

# Clinical thresholds (ADA-standard diagnostic/prediabetes cutoffs), boundary-inclusive
# on both ends, as documented in ASSESSMENT_LAYER.md. These are not invented values -
# they are the standard >=6.5% / >=126 mg/dL diabetes thresholds and the 5.7-6.4% /
# 100-125 mg/dL prediabetes range, both closed intervals.
HBA1C_DIABETIC_THRESHOLD = 6.5
HBA1C_PREDIABETIC_RANGE = (5.7, 6.4)
GLUCOSE_DIABETIC_THRESHOLD = 126
GLUCOSE_PREDIABETIC_RANGE = (100, 125)
ML_HIGH_PROBABILITY_THRESHOLD = 0.70

_model_cache = None


def _is_present(value: Optional[float]) -> bool:
    """A lab value counts as present only if it is not None and not NaN."""
    if value is None:
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def load_stage1_v2_model(path: Path = DEFAULT_MODEL_PATH, use_cache: bool = True):
    """Load the Stage 1 V2 screening pipeline from disk (read-only). Never fits
    or modifies the artifact."""
    global _model_cache
    if use_cache and _model_cache is not None:
        return _model_cache
    model = joblib.load(path)
    if use_cache:
        _model_cache = model
    return model


def extract_stage1_v2_features(form: Dict[str, Any]) -> Dict[str, Any]:
    """Extract EXACTLY the eight Stage 1 V2 features from a full assessment form.
    Raises if any required feature is absent - the pipeline cannot run without its
    eight frozen inputs. HbA1c/fasting_glucose are intentionally excluded here;
    they are never passed into the Stage 1 V2 pipeline."""
    missing = [f for f in STAGE1_V2_FEATURES if f not in form]
    if missing:
        raise ValueError(f"Missing required Stage 1 V2 feature(s): {missing}")
    return {f: form[f] for f in STAGE1_V2_FEATURES}


def _resolve_class_names(classifier) -> list:
    """Map a classifier's classes_ to display labels.

    - String classes (e.g. duck-typed test doubles, legacy artifacts) are used
      verbatim.
    - Encoded integer classes (the real Stage 1 V2 pipeline: [0, 1, 2]) are
      mapped through STAGE1_V2_CLASS_LABELS to 'High'/'Low'/'Moderate'.
    """
    raw = list(getattr(classifier, "classes_", []))
    if raw and all(isinstance(c, str) for c in raw):
        return raw
    names = []
    for c in raw:
        try:
            code = int(c)
        except (TypeError, ValueError):
            code = None
        if code is not None and code in STAGE1_V2_CLASS_LABELS:
            names.append(STAGE1_V2_CLASS_LABELS[code])
        else:
            names.append(str(c))
    return names


def _classifier_of(model):
    """Return the estimator exposing classes_ for a given model object.

    Handles real sklearn Pipelines (estimator is the final step, e.g. 'clf'),
    duck-typed test doubles that expose named_steps['classifier'], and bare
    estimators.
    """
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        return model.named_steps["classifier"]
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1]
    return model


def _label_for(value):
    """Convert a raw model output value (encoded int code or string label) into
    a display label."""
    if isinstance(value, str):
        return value
    try:
        code = int(value)
    except (TypeError, ValueError):
        return str(value)
    return STAGE1_V2_CLASS_LABELS.get(code, str(value))


def get_stage1_v2_prediction(form: Dict[str, Any], model=None) -> Tuple[str, Dict[str, float]]:
    """Run the Stage 1 V2 pipeline on the eight extracted features. Returns
    (predicted_class_label, {'High': p, 'Low': p, 'Moderate': p, ...}). HbA1c and
    fasting_glucose, even if present in `form`, are never included in the
    DataFrame passed to the pipeline."""
    model = model if model is not None else load_stage1_v2_model()
    features = extract_stage1_v2_features(form)
    X = pd.DataFrame([features])[STAGE1_V2_FEATURES]

    predicted = model.predict(X)[0]
    proba_row = model.predict_proba(X)[0]

    # predict_proba() columns follow classifier.classes_ order; resolve names
    # without relying on dictionary ordering (Rules 2/4 look up 'High'/'Low' by
    # key, never by position).
    classifier = _classifier_of(model)
    class_names = _resolve_class_names(classifier)
    proba = {name: float(p) for name, p in zip(class_names, proba_row)}

    return _label_for(predicted), proba


def assess(form: Dict[str, Any], model=None) -> Dict[str, Any]:
    """
    Apply the verified DiaBeta hybrid assessment logic.

    `form` may contain: age, sex, bmi, race_ethnicity, family_history,
    hypertension, physical_activity, smoking_status (required for the Stage 1 V2
    pipeline), and optionally hba1c, fasting_glucose (either or both may be
    missing / None / NaN).

    Rule order (evaluated top to bottom, first match wins):
      1. HbA1c >= 6.5  OR  fasting glucose >= 126                 -> HIGH
      2. ML P(High) >= 0.70 (Stage 1 V2 screener)                 -> HIGH
      3. HbA1c in [5.7, 6.4]  OR  fasting glucose in [100, 125]   -> MODERATE
      4. otherwise: ML Low/High decision (P(Low) vs P(High))       -> LOW or HIGH

    Missing-value handling: a missing (None/NaN) lab value cannot satisfy any
    condition that references it. If both labs are missing, rules 1 and 3 are
    both skipped entirely and the result depends only on the Stage 1 V2 screener
    (rules 2/4).

    Rule 4 note: the Stage 1 V2 screener is a three-class (Low/Moderate/High)
    classifier, but rule 4 is explicitly defined in the spec as a "Low/High"
    decision. Since Moderate is already fully handled by rule 3 using lab
    values, rule 4 resolves the residual case by comparing P(Low) vs P(High)
    directly (ignoring P(Moderate) for this specific tie-break) and returns
    whichever is greater. This is a documented, deliberate resolution of an
    otherwise-undefined case (a 3-class model output being asked to yield a
    2-class decision) - it is not a new clinical rule.

    Returns a dict with: final_risk_level, model1b_prediction,
    model1b_probability, rule_triggered, inputs_used (for audit/debugging).
    """
    hba1c = form.get('hba1c')
    glucose = form.get('fasting_glucose')

    hba1c_present = _is_present(hba1c)
    glucose_present = _is_present(glucose)

    ml_pred, proba = get_stage1_v2_prediction(form, model=model)
    p_high = proba.get('High', 0.0)
    p_low = proba.get('Low', 0.0)

    # ---- Rule 1: diagnostic lab override ----
    if (hba1c_present and hba1c >= HBA1C_DIABETIC_THRESHOLD) or \
       (glucose_present and glucose >= GLUCOSE_DIABETIC_THRESHOLD):
        return _build_result('High', 'rule_1_diagnostic_lab_override',
                              ml_pred, proba, hba1c, glucose)

    # ---- Rule 2: strong ML signal ----
    if p_high >= ML_HIGH_PROBABILITY_THRESHOLD:
        return _build_result('High', 'rule_2_model1b_high_probability',
                              ml_pred, proba, hba1c, glucose)

    # ---- Rule 3: prediabetic-range lab ----
    lo, hi = HBA1C_PREDIABETIC_RANGE
    glo, ghi = GLUCOSE_PREDIABETIC_RANGE
    if (hba1c_present and lo <= hba1c <= hi) or \
       (glucose_present and glo <= glucose <= ghi):
        return _build_result('Moderate', 'rule_3_prediabetic_range_lab',
                              ml_pred, proba, hba1c, glucose)

    # ---- Rule 4: ML Low/High fallback ----
    final_level = 'High' if p_high >= p_low else 'Low'
    return _build_result(final_level, 'rule_4_model1b_low_high_fallback',
                          ml_pred, proba, hba1c, glucose)


def _build_result(final_level, rule_triggered, ml_pred, proba, hba1c, glucose):
    return {
        'final_risk_level': final_level,
        'model1b_prediction': ml_pred,
        'model1b_probability': proba,
        'rule_triggered': rule_triggered,
        'inputs_used': {
            'hba1c': hba1c,
            'fasting_glucose': glucose,
            'hba1c_present': _is_present(hba1c),
            'fasting_glucose_present': _is_present(glucose),
        },
    }
