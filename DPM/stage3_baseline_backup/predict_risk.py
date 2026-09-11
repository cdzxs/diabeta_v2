"""
predict_risk.py

DiaBeta 2.0 - Stage 3 Transition Risk Prediction Module

This module loads the frozen Stage 3 artifacts:
- stage3_preprocessor.pkl: Preprocessing pipeline (numeric & categorical transformations)
- stage3_xgboost_model.pkl: XGBoost transition risk classification model
- stage3_threshold.txt: Empirically verified decision threshold (0.05)

Provides the inference function:
    predict_stage3_risk(patient_data)

Flow:
    patient data
    -> saved preprocessor
    -> saved XGBoost model
    -> transition probability
    -> threshold (0.05)
    -> result dictionary
"""

import os
from pathlib import Path
from typing import Dict, Any, Union, List, Tuple
import joblib
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Cross-version compatibility shim for scikit-learn unpickling
# (Resolves sklearn 1.6.x -> 1.7+/1.8+ internal attribute renames without
# mutating the saved frozen .pkl artifact on disk)
# ---------------------------------------------------------------------------
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, '_RemainderColsList'):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

try:
    from sklearn.impute import SimpleImputer
    _orig_imputer_transform = SimpleImputer.transform

    def _compat_imputer_transform(self, X):
        if not hasattr(self, '_fill_dtype'):
            self._fill_dtype = getattr(self, '_fit_dtype', getattr(X, 'dtype', None))
        return _orig_imputer_transform(self, X)

    SimpleImputer.transform = _compat_imputer_transform
except Exception:
    pass

# ---------------------------------------------------------------------------
# Artifact paths & required features
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_PREPROCESSOR_PATH = BASE_DIR / "stage3_preprocessor.pkl"
DEFAULT_MODEL_PATH = BASE_DIR / "stage3_xgboost_model.pkl"
DEFAULT_THRESHOLD_PATH = BASE_DIR / "stage3_threshold.txt"

# Exact 13 required features for Stage 3 prediction
REQUIRED_STAGE3_FEATURES: List[str] = [
    'sn256',
    'sx060_r',
    'sz101',
    'sz105',
    'sz106',
    'sz103',
    'sz104',
    'sz107',
    'sz108',
    'sz122',
    'sz123',
    'sz124',
    'sz080'
]

# In-memory artifact cache
_cached_preprocessor = None
_cached_model = None
_cached_threshold = None


def load_threshold(path: Path = DEFAULT_THRESHOLD_PATH) -> float:
    """Load the decision threshold from stage3_threshold.txt. Must evaluate to 0.05."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    val = float(content)
    if val != 0.05:
        raise ValueError(f"Expected threshold to be 0.05, but got {val} from {path}")
    return val


def load_artifacts(
    preprocessor_path: Path = DEFAULT_PREPROCESSOR_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    threshold_path: Path = DEFAULT_THRESHOLD_PATH,
    use_cache: bool = True
) -> Tuple[Any, Any, float]:
    """Load the frozen Stage 3 preprocessor, XGBoost model, and threshold.
    Never alters, refits, or retrains any artifact."""
    global _cached_preprocessor, _cached_model, _cached_threshold

    if use_cache and _cached_preprocessor is not None and _cached_model is not None and _cached_threshold is not None:
        return _cached_preprocessor, _cached_model, _cached_threshold

    if not preprocessor_path.is_file():
        raise FileNotFoundError(f"Stage 3 preprocessor not found at {preprocessor_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Stage 3 XGBoost model not found at {model_path}")
    if not threshold_path.is_file():
        raise FileNotFoundError(f"Stage 3 threshold file not found at {threshold_path}")

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    threshold = load_threshold(threshold_path)

    if use_cache:
        _cached_preprocessor = preprocessor
        _cached_model = model
        _cached_threshold = threshold

    return preprocessor, model, threshold


def predict_stage3_risk(patient_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    """Calculate the diabetes transition risk for a patient using the frozen Stage 3 pipeline.

    Parameters
    ----------
    patient_data : dict or pandas.DataFrame
        Must contain all 13 required Stage 3 features:
        - sn256, sx060_r, sz101, sz105, sz106, sz103, sz104,
          sz107, sz108, sz122, sz123, sz124, sz080

    Returns
    -------
    dict
        Clean result containing:
        - risk_probability (float): Model predicted probability of transition (0.0 to 1.0)
        - risk_percentage (float): Risk probability expressed as a percentage (0.0% to 100.0%)
        - transition_prediction (int): Binary prediction (1 if risk >= 0.05, else 0)
        - threshold (float): Applied decision threshold (0.05)
    """
    preprocessor, model, threshold = load_artifacts()

    # 1. Validate features and convert to DataFrame
    if isinstance(patient_data, dict):
        missing_features = [f for f in REQUIRED_STAGE3_FEATURES if f not in patient_data]
        if missing_features:
            raise ValueError(
                f"Missing required Stage 3 features: {missing_features}. "
                f"All 13 features are required: {REQUIRED_STAGE3_FEATURES}"
            )
        df_input = pd.DataFrame([{f: patient_data[f] for f in REQUIRED_STAGE3_FEATURES}])
    elif isinstance(patient_data, pd.DataFrame):
        missing_features = [f for f in REQUIRED_STAGE3_FEATURES if f not in patient_data.columns]
        if missing_features:
            raise ValueError(
                f"Missing required Stage 3 features: {missing_features}. "
                f"All 13 features are required: {REQUIRED_STAGE3_FEATURES}"
            )
        df_input = patient_data[REQUIRED_STAGE3_FEATURES].copy()
    else:
        raise TypeError(
            f"patient_data must be a dict or pandas DataFrame, got {type(patient_data).__name__}"
        )

    # 2. Use the saved Stage 3 preprocessor
    X_transformed = preprocessor.transform(df_input)

    # 3. Use the saved Stage 3 XGBoost model to calculate transition probability (class 1)
    proba = model.predict_proba(X_transformed)
    risk_prob = float(proba[0, 1])

    # 4. Apply the saved threshold of 0.05
    transition_prediction = int(risk_prob >= threshold)
    risk_percentage = round(risk_prob * 100.0, 2)

    # 5. Return clean dictionary
    return {
        "risk_probability": risk_prob,
        "risk_percentage": risk_percentage,
        "transition_prediction": transition_prediction,
        "threshold": threshold,
    }


if __name__ == "__main__":
    print("=== Stage 3 Risk Prediction Smoke Test ===")
    sample_patient = {
        'sn256': 55.0,
        'sx060_r': 1,
        'sz101': 1,
        'sz105': 0,
        'sz106': 0,
        'sz103': 1,
        'sz104': 0,
        'sz107': 0,
        'sz108': 1,
        'sz122': 0,
        'sz123': 0,
        'sz124': 0,
        'sz080': 1,
    }
    result = predict_stage3_risk(sample_patient)
    print("Sample Patient Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
