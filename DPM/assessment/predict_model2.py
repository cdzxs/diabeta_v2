"""
predict_model2.py

DiaBeta 2.0 - Stage 2 Incident Diabetes Risk Prediction Module (Model 2)

Loads the frozen Model 2 artifact (DPM/models/diabeta_dataset2_model.pkl) - a
standard (non-class-weighted) Logistic Regression pipeline trained on the
seven-predictor HRS-derived Dataset 2 (see MODEL_CARD.md). This module contains
no machine learning of its own; it only loads the frozen pipeline read-only and
exposes a clean inference interface.

Model 2 pipeline (frozen artifact):
    preprocessor: ColumnTransformer
        - BMI:            SimpleImputer(median, add_indicator=True)  -> missingness
                          preserved as an explicit indicator column
        - other_medical:  SimpleImputer(median) on Hypertension / High_Cholesterol /
                          Heart_Disease / Stroke
        - demographic:    SimpleImputer(median) on Age / Sex
    scaler:      StandardScaler
    classifier:  LogisticRegression(max_iter=1000, random_state=42)

Exactly seven predictors, in this order: Age, Sex, BMI (may be missing),
Hypertension, High_Cholesterol, Heart_Disease, Stroke. Participant_ID is an
identifier only and is never passed to the model.

Output: P(class 1) = probability of incident diabetes. There is no validated
clinical threshold, so only the calibrated probability is returned - never a
binary class label.

See MODEL_CARD.md (project root) for the full model card.
"""

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'diabeta_dataset2_model.pkl'

# The seven, and only seven, predictors the frozen Model 2 pipeline accepts.
MODEL2_FEATURES = [
    'Age',
    'Sex',
    'BMI',
    'Hypertension',
    'High_Cholesterol',
    'Heart_Disease',
    'Stroke',
]

_model_cache = None


def load_model2(path: Path = DEFAULT_MODEL_PATH, use_cache: bool = True):
    """Load the frozen Model 2 pipeline from disk (read-only). Never fits or
    modifies the artifact."""
    global _model_cache
    if use_cache and _model_cache is not None:
        return _model_cache
    if not path.is_file():
        raise FileNotFoundError(f"Model 2 artifact not found at {path}")
    model = joblib.load(path)
    if use_cache:
        _model_cache = model
    return model


def _map_sex(value: Any) -> int:
    """Map a raw sex value to Model 2's encoding (0 = Female, 1 = Male).

    Accepts the Stage 2 form encoding (0/1), the Stage 1 / HRS numeric coding
    (1 = Male, 2 = Female), and common string forms.
    """
    if value is None:
        raise ValueError("Invalid value for Sex: expected Male or Female.")
    s = str(value).strip().lower()
    if s in ('1', '1.0', 'male', 'm'):
        return 1
    if s in ('0', '0.0', '2', '2.0', 'female', 'f'):
        return 0
    raise ValueError("Invalid value for Sex: expected Male or Female.")


def _map_binary(value: Any, name: str) -> int:
    """Map a raw 0/1 value to an int for a binary condition field."""
    if value is None:
        raise ValueError(f"Invalid value for {name}: expected 0 (No) or 1 (Yes).")
    s = str(value).strip().lower()
    if s in ('1', '1.0', 'true', 'yes'):
        return 1
    if s in ('0', '0.0', 'false', 'no'):
        return 0
    raise ValueError(f"Invalid value for {name}: expected 0 (No) or 1 (Yes).")


def _is_positive_number(value: Any, name: str) -> float:
    """Coerce a numeric field to float, rejecting non-numeric / non-positive values."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid value for {name}: must be a numeric value.")
    if math.isnan(val):
        return val  # BMI may be missing; the pipeline imputes it
    if val <= 0:
        raise ValueError(f"Invalid value for {name}: must be a positive number.")
    return val


def predict_stage2_risk(patient_data: Dict[str, Any], model=None) -> Dict[str, Any]:
    """Calculate the incident-diabetes probability for a patient using the frozen
    Model 2 pipeline.

    Parameters
    ----------
    patient_data : dict
        Must contain the 7 Model 2 predictors under these keys: Age, Sex, BMI,
        Hypertension, High_Cholesterol, Heart_Disease, Stroke. BMI may be
        missing (None / NaN) - the frozen pipeline's median imputation plus
        missingness indicator handles it. Extra keys (e.g. patient_name) are
        ignored and never passed to the model.

    Returns
    -------
    dict
        - risk_probability (float): P(incident diabetes) from the frozen model (0-1)
        - risk_percentage (float):  risk_probability expressed as a percentage
        - features (dict): the evaluated 7-factor profile (for display)
    """
    model = model if model is not None else load_model2()

    # 1. Validate that every required feature is present
    missing = [f for f in MODEL2_FEATURES if f not in patient_data]
    if missing:
        raise ValueError(
            f"Missing required Model 2 feature(s): {missing}. "
            f"All 7 features are required: {MODEL2_FEATURES}"
        )

    # 2. Normalize / validate the individual values
    age = _is_positive_number(patient_data['Age'], 'Age')
    sex = _map_sex(patient_data['Sex'])
    bmi_raw = patient_data['BMI']
    if bmi_raw is None or (isinstance(bmi_raw, float) and math.isnan(bmi_raw)) \
            or str(bmi_raw).strip().lower() in ('', 'null', 'nan'):
        bmi = float('nan')
    else:
        bmi = _is_positive_number(bmi_raw, 'BMI')
    hyp = _map_binary(patient_data['Hypertension'], 'Hypertension')
    chol = _map_binary(patient_data['High_Cholesterol'], 'High_Cholesterol')
    heart = _map_binary(patient_data['Heart_Disease'], 'Heart_Disease')
    stroke = _map_binary(patient_data['Stroke'], 'Stroke')

    row = {
        'Age': age,
        'Sex': float(sex),
        'BMI': bmi,
        'Hypertension': float(hyp),
        'High_Cholesterol': float(chol),
        'Heart_Disease': float(heart),
        'Stroke': float(stroke),
    }
    X = pd.DataFrame([row])[MODEL2_FEATURES]

    # 3. Run the frozen pipeline (preprocessor -> scaler -> logistic regression)
    risk_prob = float(model.predict_proba(X)[0, 1])
    risk_percentage = round(risk_prob * 100.0, 2)

    # 4. Return a clean, JSON-serializable result
    return {
        "risk_probability": round(risk_prob, 6),
        "risk_percentage": risk_percentage,
        "features": {
            "Age": age,
            "Sex": sex,
            "BMI": None if math.isnan(bmi) else bmi,
            "Hypertension": hyp,
            "High_Cholesterol": chol,
            "Heart_Disease": heart,
            "Stroke": stroke,
        },
    }


if __name__ == "__main__":
    print("=== Stage 2 (Model 2) Risk Prediction Smoke Test ===")
    sample_patient = {
        'Age': 58.0,
        'Sex': 1,
        'BMI': float('nan'),
        'Hypertension': 1,
        'High_Cholesterol': 0,
        'Heart_Disease': 1,
        'Stroke': 0,
    }
    result = predict_stage2_risk(sample_patient)
    print("Sample Patient Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")