from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "output" / "unified_rich_3y_model.joblib"


class OutsideEvidenceError(ValueError):
    pass


class NotApplicableError(ValueError):
    pass


def _male(sex: str | int) -> int:
    if isinstance(sex, str):
        value = sex.strip().lower()
        if value in {"male", "m", "man"}:
            return 1
        if value in {"female", "f", "woman"}:
            return 0
    if sex in {0, 1}:
        return int(sex)
    raise ValueError("sex must be male/female or 1/0")


def predict(profile: dict[str, Any], model_path: Path = MODEL_PATH):
    age = float(profile["age"])
    bmi = float(profile["bmi"])
    fpg = float(profile["fpg_mg_dl"])
    hba1c = profile.get("hba1c_pct")
    if fpg >= 126 or (hba1c is not None and float(hba1c) >= 6.5):
        raise NotApplicableError(
            "A diabetes-range laboratory result requires clinical confirmation; "
            "future-risk scoring is not appropriate."
        )
    if not 50.45 <= fpg < 126:
        raise OutsideEvidenceError("Fasting glucose is outside the supported model range.")
    if not 20 <= age <= 99:
        raise OutsideEvidenceError("Age is outside the observed 20–99 year range.")
    if not 15 <= bmi <= 52.7:
        raise OutsideEvidenceError("BMI is outside the observed model range.")

    warnings = []
    if age < 25 or age > 78 or bmi < 17 or bmi > 32.2 or fpg < 62 or fpg > 117.65:
        warnings.append("One or more values are in a sparse outer 1% of the training cohort.")
    x = np.array([[age, _male(profile["sex"]), bmi, fpg / 18.0182]])
    fitted = joblib.load(model_path)
    risk = float(fitted.predict_proba(x)[0, 1])
    return {
        "model_used": "unified_rich_3y",
        "risk_probability": {"3_year": risk},
        "risk_percent": {"3_year": round(risk * 100, 2)},
        "reliability": "caution" if warnings else "supported",
        "warnings": warnings,
        "interpretation": "Estimated probability of developing diabetes by approximately three years.",
    }
