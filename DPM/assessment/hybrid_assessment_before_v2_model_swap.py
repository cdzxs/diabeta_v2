"""
hybrid_assessment.py

DiaBeta 2.0 - Assessment Layer (Stage 2)

Combines the FROZEN Model 1B screening pipeline (Dataset 1 / NHANES, five
non-diagnostic features) with the verified original hybrid rule set that
incorporates HbA1c and fasting glucose. This module contains NO machine
learning of its own - it is deterministic Python logic that calls the frozen
Model 1B pipeline and applies clinical-threshold rules on top of its output.

Model 1B is loaded read-only from disk and is never fitted, refit, or altered
by this module.
"""

import math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import joblib
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'diabeta_dataset1_model.pkl'

# The five, and only five, features the frozen Model 1B pipeline accepts.
MODEL1B_FEATURES = ['age', 'bmi', 'sex', 'race_ethnicity', 'family_history']

# Clinical thresholds (ADA-standard diagnostic/prediabetes cutoffs), boundary-inclusive
# on both ends, as documented in ASSESSMENT_LAYER.md. These are not invented values -
# they are the standard >=6.5% / >=126 mg/dL diabetes thresholds and the 5.7-6.4% /
# 100-125 mg/dL prediabetes range, both closed intervals.
HBA1C_DIABETIC_THRESHOLD = 6.5
HBA1C_PREDIABETIC_RANGE = (5.7, 6.4)
GLUCOSE_DIABETIC_THRESHOLD = 126
GLUCOSE_PREDIABETIC_RANGE = (100, 125)
MODEL1B_HIGH_PROBABILITY_THRESHOLD = 0.70

_model_cache = None


def _is_present(value: Optional[float]) -> bool:
    """A lab value counts as present only if it is not None and not NaN."""
    if value is None:
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def load_model1b(path: Path = DEFAULT_MODEL_PATH, use_cache: bool = True):
    """Load the frozen Model 1B pipeline from disk (read-only). Never fits or
    modifies the artifact."""
    global _model_cache
    if use_cache and _model_cache is not None:
        return _model_cache
    model = joblib.load(path)
    if use_cache:
        _model_cache = model
    return model


def extract_model1b_features(form: Dict[str, Any]) -> Dict[str, Any]:
    """Extract EXACTLY the five Model 1B features from a full assessment form.
    Raises if any required Model 1B feature is absent - Model 1B cannot run
    without its five frozen inputs. HbA1c/fasting_glucose are intentionally
    excluded here; they are never passed into Model 1B."""
    missing = [f for f in MODEL1B_FEATURES if f not in form]
    if missing:
        raise ValueError(f"Missing required Model 1B feature(s): {missing}")
    return {f: form[f] for f in MODEL1B_FEATURES}


def get_model1b_prediction(form: Dict[str, Any], model=None) -> Tuple[str, Dict[str, float]]:
    """Run the frozen Model 1B pipeline on the five extracted features. Returns
    (predicted_class, {class_name: probability, ...}). HbA1c and fasting_glucose,
    even if present in `form`, are never included in the DataFrame passed to
    Model 1B."""
    model = model if model is not None else load_model1b()
    features = extract_model1b_features(form)
    X = pd.DataFrame([features])[MODEL1B_FEATURES]

    predicted_class = model.predict(X)[0]
    proba_row = model.predict_proba(X)[0]
    classifier = model.named_steps['classifier']
    classes = list(classifier.classes_)
    proba = {cls: float(p) for cls, p in zip(classes, proba_row)}

    return predicted_class, proba


def assess(form: Dict[str, Any], model=None) -> Dict[str, Any]:
    """
    Apply the verified DiaBeta hybrid assessment logic.

    `form` may contain: age, bmi, sex, race_ethnicity, family_history (required
    for Model 1B), and optionally hba1c, fasting_glucose (either or both may be
    missing / None / NaN).

    Rule order (evaluated top to bottom, first match wins):
      1. HbA1c >= 6.5  OR  fasting glucose >= 126           -> HIGH
      2. Model 1B P(High) >= 0.70                            -> HIGH
      3. HbA1c in [5.7, 6.4]  OR  fasting glucose in [100,125] -> MODERATE
      4. otherwise: Model 1B's Low/High decision (P(Low) vs P(High)) -> LOW or HIGH

    Missing-value handling: a missing (None/NaN) lab value cannot satisfy any
    condition that references it. If both labs are missing, rules 1 and 3 are
    both skipped entirely and the result depends only on Model 1B (rules 2/4).

    Rule 4 note: the frozen Model 1B is a three-class (Low/Moderate/High)
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

    model1b_pred, proba = get_model1b_prediction(form, model=model)
    p_high = proba.get('High', 0.0)
    p_low = proba.get('Low', 0.0)

    # ---- Rule 1: diagnostic lab override ----
    if (hba1c_present and hba1c >= HBA1C_DIABETIC_THRESHOLD) or \
       (glucose_present and glucose >= GLUCOSE_DIABETIC_THRESHOLD):
        return _build_result('High', 'rule_1_diagnostic_lab_override',
                              model1b_pred, proba, hba1c, glucose)

    # ---- Rule 2: strong Model 1B signal ----
    if p_high >= MODEL1B_HIGH_PROBABILITY_THRESHOLD:
        return _build_result('High', 'rule_2_model1b_high_probability',
                              model1b_pred, proba, hba1c, glucose)

    # ---- Rule 3: prediabetic-range lab ----
    lo, hi = HBA1C_PREDIABETIC_RANGE
    glo, ghi = GLUCOSE_PREDIABETIC_RANGE
    if (hba1c_present and lo <= hba1c <= hi) or \
       (glucose_present and glo <= glucose <= ghi):
        return _build_result('Moderate', 'rule_3_prediabetic_range_lab',
                              model1b_pred, proba, hba1c, glucose)

    # ---- Rule 4: Model 1B Low/High fallback ----
    final_level = 'High' if p_high >= p_low else 'Low'
    return _build_result(final_level, 'rule_4_model1b_low_high_fallback',
                          model1b_pred, proba, hba1c, glucose)


def _build_result(final_level, rule_triggered, model1b_pred, proba, hba1c, glucose):
    return {
        'final_risk_level': final_level,
        'model1b_prediction': model1b_pred,
        'model1b_probability': proba,
        'rule_triggered': rule_triggered,
        'inputs_used': {
            'hba1c': hba1c,
            'fasting_glucose': glucose,
            'hba1c_present': _is_present(hba1c),
            'fasting_glucose_present': _is_present(glucose),
        },
    }
