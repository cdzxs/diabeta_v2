import os
import sys
import math
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
if __package__:
    from .stage2_explanation import explain as explain_stage2
else:
    from stage2_explanation import explain as explain_stage2

app = Flask(__name__)
CORS(app)  # allow frontend (different domain) to call the API

BASE_DIR = Path(__file__).resolve().parent
DPM_ASSESSMENT_DIR = BASE_DIR.parent / "DPM" / "assessment"
if str(DPM_ASSESSMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DPM_ASSESSMENT_DIR))

DPM_STAGE3_UNIFIED_DIR = BASE_DIR.parent / "DPM" / "stage3" / "unified_rich_model"
if str(DPM_STAGE3_UNIFIED_DIR) not in sys.path:
    sys.path.insert(0, str(DPM_STAGE3_UNIFIED_DIR))

import hybrid_assessment as ha
import inference as unified_s3
from inference import NotApplicableError as S3NotApplicableError, OutsideEvidenceError as S3OutsideEvidenceError

# Warm up / verify the unified Stage 3 Rich 3-year model at startup
_unified_s3_model = joblib.load(unified_s3.MODEL_PATH)


# ---- Stage 1 V2 screener (validated 8-predictor, non-laboratory model) ----
# Active Stage 1 model: DPM/models/stage1_v2_model.pkl
# (validated HistGradientBoosting pipeline; see DPM/models/stage1_v2_training_report.txt)
STAGE1_MODEL_PATH = BASE_DIR.parent / "DPM" / "models" / "stage1_v2_model.pkl"
STAGE1_FEATURES = [
    "age", "sex", "bmi", "race_ethnicity",
    "family_history", "hypertension", "physical_activity", "smoking_status",
]
# The pipeline was trained on LabelEncoder-encoded targets (classes sorted
# alphabetically: 0=High, 1=Low, 2=Moderate). Verify that contract at load.
STAGE1_CLASS_LABELS = ["High", "Low", "Moderate"]
stage1_model = joblib.load(STAGE1_MODEL_PATH)
if list(stage1_model.classes_) != [0, 1, 2]:
    raise RuntimeError(
        f"Unexpected Stage 1 model classes {list(stage1_model.classes_)} - "
        f"expected LabelEncoder classes [0, 1, 2] (High, Low, Moderate)."
    )

# Load race mapping for /api/race-map and frontend display labels
race_map_file = BASE_DIR / "model" / "race_map.pkl"
if race_map_file.is_file():
    race_map = joblib.load(race_map_file)
else:
    race_map = {
        1: "Mexican American",
        2: "Other Hispanic",
        3: "Non-Hispanic White",
        4: "Non-Hispanic Black",
        6: "Non-Hispanic Asian",
        7: "Other / Multi-racial"
    }


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/api/race-map")
def get_race_map():
    return jsonify(race_map)


@app.route("/api/predict", methods=["POST"])
def predict():
    """Stage 1 - Screener (validated 8-predictor, non-laboratory model).

    Accepts EXACTLY the 8 locked Stage 1 predictors:
        age, sex, bmi, race_ethnicity, family_history,
        hypertension, physical_activity, smoking_status
    No laboratory values are accepted here (HbA1c / fasting glucose belong
    to Stage 2, /api/predict-stage2).
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid request body: expected a JSON object."}), 400

    # ---- 1. Validate & parse the eight required Stage 1 inputs ----
    # age
    if "age" not in data or data["age"] is None or str(data["age"]).strip() == "":
        return jsonify({"error": "Missing required field: age"}), 400
    try:
        age = float(data["age"])
        if isinstance(data["age"], bool) or age <= 0 or age > 130 or not math.isfinite(age):
            return jsonify({"error": "Invalid value for age: must be a positive number up to 130."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid value for age: must be a numeric value."}), 400

    # bmi
    if "bmi" not in data or data["bmi"] is None or str(data["bmi"]).strip() == "":
        return jsonify({"error": "Missing required field: bmi"}), 400
    try:
        bmi = float(data["bmi"])
        if isinstance(data["bmi"], bool) or bmi <= 0 or bmi > 150 or not math.isfinite(bmi):
            return jsonify({"error": "Invalid value for bmi: must be a positive number."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid value for bmi: must be a numeric value."}), 400

    # sex (frontend 1 = Male / 0 = Female -> NHANES 1 = Male, 2 = Female)
    if "sex" not in data or data["sex"] is None or str(data["sex"]).strip() == "":
        return jsonify({"error": "Missing required field: sex"}), 400
    raw_sex = str(data["sex"]).strip().lower()
    if raw_sex in ("1", "1.0", "male", "m"):
        sex_mapped = 1.0
        sex_label = "Male"
    elif raw_sex in ("0", "0.0", "2", "2.0", "female", "f"):
        sex_mapped = 2.0
        sex_label = "Female"
    else:
        return jsonify({"error": "Invalid value for sex: expected 1 (Male) or 0/2 (Female)."}), 400

    # race_ethnicity
    if "race_ethnicity" not in data or data["race_ethnicity"] is None or str(data["race_ethnicity"]).strip() == "":
        return jsonify({"error": "Missing required field: race_ethnicity"}), 400
    try:
        race_value = float(data["race_ethnicity"])
        if isinstance(data["race_ethnicity"], bool) or race_value not in (1, 2, 3, 4, 6, 7):
            raise ValueError("Unsupported race code")
        race_code = int(race_value)
    except (ValueError, TypeError, OverflowError):
        return jsonify({"error": "Invalid value for race_ethnicity: must be an integer code."}), 400
    race_label = race_map.get(race_code, race_map.get(str(race_code), f"Code {race_code}"))

    # family_history (1 -> 1.0 Yes, 0 -> 0.0 No, unknown -> NaN imputed by pipeline)
    if "family_history" not in data or data["family_history"] is None or str(data["family_history"]).strip() == "":
        return jsonify({"error": "Missing required field: family_history"}), 400
    raw_fam = str(data["family_history"]).strip().lower()
    if raw_fam in ("1", "1.0", "yes", "true"):
        fam_mapped = 1.0
        fam_label = "Yes"
    elif raw_fam in ("0", "0.0", "no", "false"):
        fam_mapped = 0.0
        fam_label = "No"
    elif raw_fam in ("unknown", "-1", "null"):
        fam_mapped = float("nan")
        fam_label = "Unknown"
    else:
        return jsonify({"error": "Invalid value for family_history: expected 1 (Yes) or 0 (No)."}), 400

    # hypertension (1 -> 1.0 Yes, 0 -> 0.0 No)
    if "hypertension" not in data or data["hypertension"] is None or str(data["hypertension"]).strip() == "":
        return jsonify({"error": "Missing required field: hypertension"}), 400
    raw_hyp = str(data["hypertension"]).strip().lower()
    if raw_hyp in ("1", "1.0", "yes", "true"):
        hyp_mapped = 1.0
        hyp_label = "Yes"
    elif raw_hyp in ("0", "0.0", "no", "false"):
        hyp_mapped = 0.0
        hyp_label = "No"
    else:
        return jsonify({"error": "Invalid value for hypertension: expected 1 (Yes) or 0 (No)."}), 400

    # physical_activity (1 -> 1.0 Active, 0 -> 0.0 Inactive)
    if "physical_activity" not in data or data["physical_activity"] is None or str(data["physical_activity"]).strip() == "":
        return jsonify({"error": "Missing required field: physical_activity"}), 400
    raw_pa = str(data["physical_activity"]).strip().lower()
    if raw_pa in ("1", "1.0", "yes", "true", "active"):
        pa_mapped = 1.0
        pa_label = "Active"
    elif raw_pa in ("0", "0.0", "no", "false", "inactive"):
        pa_mapped = 0.0
        pa_label = "Inactive"
    else:
        return jsonify({"error": "Invalid value for physical_activity: expected 1 (Active) or 0 (Inactive)."}), 400

    # smoking_status (0 = Never, 1 = Former, 2 = Current)
    if "smoking_status" not in data or data["smoking_status"] is None or str(data["smoking_status"]).strip() == "":
        return jsonify({"error": "Missing required field: smoking_status"}), 400
    raw_smk = str(data["smoking_status"]).strip().lower()
    if raw_smk in ("0", "0.0", "never"):
        smk_mapped = 0.0
        smk_label = "Never"
    elif raw_smk in ("1", "1.0", "former"):
        smk_mapped = 1.0
        smk_label = "Former"
    elif raw_smk in ("2", "2.0", "current"):
        smk_mapped = 2.0
        smk_label = "Current"
    else:
        return jsonify({"error": "Invalid value for smoking_status: expected 0 (Never), 1 (Former), or 2 (Current)."}), 400

    patient_name = str(data.get("patient_name", "")).strip()
    patient_address = str(data.get("patient_address", "")).strip()

    # ---- 2. Build the exact 8-predictor row for the validated pipeline ----
    row = pd.DataFrame([{
        "age": age,
        "sex": sex_mapped,
        "bmi": bmi,
        "race_ethnicity": float(race_code),
        "family_history": fam_mapped,
        "hypertension": hyp_mapped,
        "physical_activity": pa_mapped,
        "smoking_status": smk_mapped,
    }])[STAGE1_FEATURES]

    # ---- 3. Run the validated Stage 1 V2 screener ----
    try:
        code = int(stage1_model.predict(row)[0])
        proba = stage1_model.predict_proba(row)[0]
    except Exception as e:
        return jsonify({"error": f"Stage 1 prediction error: {str(e)}"}), 500

    # Encoded classes: 0=High, 1=Low, 2=Moderate (LabelEncoder contract)
    tier = STAGE1_CLASS_LABELS[code]
    p_high = float(proba[0])
    prob_high = float(round(p_high * 100, 1))

    # ---- 4. User-friendly reason ----
    reasons = [
        f"Stage 1 screener assessed 8 non-laboratory factors (age, sex, BMI, "
        f"race/ethnicity, family history, hypertension, physical activity, "
        f"smoking status). Model assigned {tier} risk with a {prob_high}% "
        f"probability of a high-risk diabetes profile."
    ]

    # ---- 5. Person summary (preserves Stage 1 -> Stage 2 handoff keys) ----
    person_summary = {
        "Patient Name": patient_name if patient_name else "Not provided",
        "Patient Address": patient_address if patient_address else "Not provided",
        "Age": age,
        "Sex": sex_label,
        "BMI": bmi,
        "Race/Ethnicity": race_label,
        "Family History": fam_label,
        "Hypertension": hyp_label,
        "Physical Activity": pa_label,
        "Smoking Status": smk_label,
    }

    return jsonify({
        "tier": tier,
        "prob_high": prob_high,
        "reason": reasons,
        "shap_explanation": None,
        "person": person_summary
    }), 200


@app.route("/api/predict-stage2", methods=["POST"])
def predict_stage2():
    """Stage 2 - Detailed Risk Assessment (hybrid assessment layer).

    Validates the eight assessment inputs (age, sex, bmi, race_ethnicity,
    family_history, hypertension, physical_activity, smoking_status), accepts
    optional patient identity fields (patient_name, patient_address - used for
    the person summary display only, never as model features), accepts
    HbA1c and fasting glucose (at least one must be supplied), and runs
    the existing hybrid assessment engine
    (DPM/assessment/hybrid_assessment.py) on top of the validated Stage 1 V2
    screener.
    """
    return assess_stage2_request(request.get_json(silent=True))


def assess_stage2_request(data):
    """Shared validation and server-side assessment for Stages 2 and 3."""
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid request body: expected a JSON object."}), 400

    # 1. Validate & parse the eight Stage 1 screener inputs
    # age
    if "age" not in data or data["age"] is None or str(data["age"]).strip() == "":
        return jsonify({"error": "Missing required field: age"}), 400
    try:
        age = float(data["age"])
        if isinstance(data["age"], bool) or age <= 0 or age > 130 or not math.isfinite(age):
            return jsonify({"error": "Invalid value for age: must be a positive number up to 130."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid value for age: must be a numeric value."}), 400

    # bmi
    if "bmi" not in data or data["bmi"] is None or str(data["bmi"]).strip() == "":
        return jsonify({"error": "Missing required field: bmi"}), 400
    try:
        bmi = float(data["bmi"])
        if isinstance(data["bmi"], bool) or bmi <= 0 or bmi > 150 or not math.isfinite(bmi):
            return jsonify({"error": "Invalid value for bmi: must be a positive number."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid value for bmi: must be a numeric value."}), 400

    # sex (1 -> 1.0 Male, 0 or 2 -> 2.0 Female, matching Stage 1 semantics)
    if "sex" not in data or data["sex"] is None or str(data["sex"]).strip() == "":
        return jsonify({"error": "Missing required field: sex"}), 400
    raw_sex = str(data["sex"]).strip().lower()
    if raw_sex in ("1", "1.0", "male", "m"):
        sex_mapped = 1.0
        sex_label = "Male"
    elif raw_sex in ("0", "0.0", "2", "2.0", "female", "f"):
        sex_mapped = 2.0
        sex_label = "Female"
    else:
        return jsonify({"error": "Invalid value for sex: expected 1 (Male) or 0/2 (Female)."}), 400

    # race_ethnicity
    if "race_ethnicity" not in data or data["race_ethnicity"] is None or str(data["race_ethnicity"]).strip() == "":
        return jsonify({"error": "Missing required field: race_ethnicity"}), 400
    try:
        race_value = float(data["race_ethnicity"])
        if isinstance(data["race_ethnicity"], bool) or race_value not in (1, 2, 3, 4, 6, 7):
            raise ValueError("Unsupported race code")
        race_code = int(race_value)
    except (ValueError, TypeError, OverflowError):
        return jsonify({"error": "Invalid value for race_ethnicity: must be an integer code."}), 400
    race_label = race_map.get(race_code, race_map.get(str(race_code), f"Code {race_code}"))

    # family_history (1 -> 1.0, 0 -> 0.0, unknown -> NaN)
    if "family_history" not in data or data["family_history"] is None or str(data["family_history"]).strip() == "" or str(data["family_history"]).strip().lower() == "null":
        return jsonify({"error": "Missing required field: family_history"}), 400
    raw_fam = str(data["family_history"]).strip().lower()
    if raw_fam in ("1", "1.0", "yes", "true"):
        fam_mapped = 1.0
        fam_label = "Yes"
    elif raw_fam in ("0", "0.0", "no", "false"):
        fam_mapped = 0.0
        fam_label = "No"
    elif raw_fam in ("unknown", "-1"):
        fam_mapped = float("nan")
        fam_label = "Unknown"
    else:
        return jsonify({"error": "Invalid value for family_history: expected 1 (Yes), 0 (No), or unknown."}), 400

    # hypertension (1 -> 1.0 Yes, 0 -> 0.0 No)
    if "hypertension" not in data or data["hypertension"] is None or str(data["hypertension"]).strip() == "":
        return jsonify({"error": "Missing required field: hypertension"}), 400
    raw_hyp = str(data["hypertension"]).strip().lower()
    if raw_hyp in ("1", "1.0", "yes", "true"):
        hyp_mapped = 1.0
        hyp_label = "Yes"
    elif raw_hyp in ("0", "0.0", "no", "false"):
        hyp_mapped = 0.0
        hyp_label = "No"
    else:
        return jsonify({"error": "Invalid value for hypertension: expected 1 (Yes) or 0 (No)."}), 400

    # physical_activity (1 -> 1.0 Active, 0 -> 0.0 Inactive)
    if "physical_activity" not in data or data["physical_activity"] is None or str(data["physical_activity"]).strip() == "":
        return jsonify({"error": "Missing required field: physical_activity"}), 400
    raw_pa = str(data["physical_activity"]).strip().lower()
    if raw_pa in ("1", "1.0", "yes", "true", "active"):
        pa_mapped = 1.0
        pa_label = "Active"
    elif raw_pa in ("0", "0.0", "no", "false", "inactive"):
        pa_mapped = 0.0
        pa_label = "Inactive"
    else:
        return jsonify({"error": "Invalid value for physical_activity: expected 1 (Active) or 0 (Inactive)."}), 400

    # smoking_status (0 = Never, 1 = Former, 2 = Current)
    if "smoking_status" not in data or data["smoking_status"] is None or str(data["smoking_status"]).strip() == "":
        return jsonify({"error": "Missing required field: smoking_status"}), 400
    raw_smk = str(data["smoking_status"]).strip().lower()
    if raw_smk in ("0", "0.0", "never"):
        smk_mapped = 0.0
        smk_label = "Never"
    elif raw_smk in ("1", "1.0", "former"):
        smk_mapped = 1.0
        smk_label = "Former"
    elif raw_smk in ("2", "2.0", "current"):
        smk_mapped = 2.0
        smk_label = "Current"
    else:
        return jsonify({"error": "Invalid value for smoking_status: expected 0 (Never), 1 (Former), or 2 (Current)."}), 400

    # 2. Parse optional laboratory values (either, both, or neither may be supplied)
    # HbA1c
    hba1c = None
    raw_hba1c = data.get("hba1c")
    if raw_hba1c is not None and str(raw_hba1c).strip() != "" and str(raw_hba1c).strip().lower() != "null":
        try:
            hba1c = float(raw_hba1c)
            if isinstance(raw_hba1c, bool) or not math.isfinite(hba1c) or hba1c <= 0:
                return jsonify({"error": "Invalid value for hba1c: must be a positive number."}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid value for hba1c: must be a numeric value."}), 400

    # Fasting glucose (accepts fasting_glucose_mgdl or fasting_glucose)
    glucose = None
    raw_glucose = data.get("fasting_glucose_mgdl") if "fasting_glucose_mgdl" in data else data.get("fasting_glucose")
    if raw_glucose is not None and str(raw_glucose).strip() != "" and str(raw_glucose).strip().lower() != "null":
        try:
            glucose = float(raw_glucose)
            if isinstance(raw_glucose, bool) or not math.isfinite(glucose) or glucose <= 0:
                return jsonify({"error": "Invalid value for fasting glucose: must be a positive number."}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid value for fasting glucose: must be a numeric value."}), 400

    # 3. At least one laboratory value (HbA1c or fasting glucose) is required
    #    for the Stage 2 assessment
    if hba1c is None and glucose is None:
        return jsonify({
            "error": "Stage 2 requires at least one laboratory value: provide HbA1c, fasting glucose, or both."
        }), 400

    # 4. Assemble assessment input (five screener inputs + labs)
    form_input = {
        "age": age,
        "bmi": bmi,
        "sex": sex_mapped,
        "race_ethnicity": float(race_code),
        "family_history": fam_mapped,
        "hypertension": hyp_mapped,
        "physical_activity": pa_mapped,
        "smoking_status": smk_mapped,
        "hba1c": hba1c,
        "fasting_glucose": glucose,
    }

    # 5. Run the existing hybrid assessment engine (deterministic rules over the
    #    validated Stage 1 V2 screener). Rules, thresholds, and order are unchanged.
    try:
        assessment = ha.assess(form_input, model=stage1_model)
    except Exception as e:
        return jsonify({"error": f"Assessment execution error: {str(e)}"}), 500

    tier = assessment["final_risk_level"]
    # Opt-in local diagnostics: assessment features only, never patient identity.
    if os.environ.get("DIABETA_TRACE") == "1":
        app.logger.warning("DIABETA_TRACE path=%s inputs=%s tier=%s rule=%s probabilities=%s",
                           request.path, form_input, tier, assessment["rule_triggered"],
                           assessment["model1b_probability"])
    prob_high = float(round(float(assessment["model1b_probability"].get("High", 0.0)) * 100, 1))
    rule_triggered = assessment["rule_triggered"]

    # 6. Map rule_triggered to clear human-readable reasons (never expose the
    #    internal rule identifiers to end users)
    reasons = []
    if rule_triggered == "rule_1_diagnostic_lab_override":
        if hba1c is not None and hba1c >= ha.HBA1C_DIABETIC_THRESHOLD:
            reasons.append(f"HbA1c {hba1c}% meets or exceeds the clinical diabetes threshold (≥{ha.HBA1C_DIABETIC_THRESHOLD}%).")
        if glucose is not None and glucose >= ha.GLUCOSE_DIABETIC_THRESHOLD:
            reasons.append(f"Fasting glucose {glucose} mg/dL meets or exceeds the clinical diabetes threshold (≥{ha.GLUCOSE_DIABETIC_THRESHOLD} mg/dL).")
        if not reasons:
            reasons.append("Clinical lab override applied based on diagnostic laboratory values.")
    elif rule_triggered == "rule_2_model1b_high_probability":
        reasons.append(f"The ML screener identified a high-risk profile with {prob_high}% probability based on non-laboratory factors.")
    elif rule_triggered == "rule_3_prediabetic_range_lab":
        lo, hi = ha.HBA1C_PREDIABETIC_RANGE
        glo, ghi = ha.GLUCOSE_PREDIABETIC_RANGE
        if hba1c is not None and lo <= hba1c <= hi:
            reasons.append(f"HbA1c {hba1c}% is within the clinical prediabetes range ({lo}–{hi}%).")
        if glucose is not None and glo <= glucose <= ghi:
            reasons.append(f"Fasting glucose {glucose} mg/dL is within the clinical prediabetes range ({glo}–{ghi} mg/dL).")
        if not reasons:
            reasons.append("Clinical prediabetes threshold criteria met based on laboratory values.")
    elif rule_triggered == "rule_4_model1b_low_high_fallback":
        if hba1c is None and glucose is None:
            reasons.append("Assessment based on the ML screener using demographic and clinical factors (no laboratory values provided).")
        else:
            reasons.append("No earlier rule applied to the supplied laboratory values; the profile-model fallback determined the recommendation. Missing tests were not assessed.")
    else:
        reasons.append(f"Assessment determined by the hybrid clinical rules.")

    # 7. Person summary (same shape as Stage 1, for cross-stage data transfer).
    #    Patient identity fields are display-only: they are NOT part of
    #    form_input above and never reach the hybrid assessment or any model.
    patient_name = str(data.get("patient_name", "")).strip()
    patient_address = str(data.get("patient_address", "")).strip()
    person_summary = {
        "Patient Name": patient_name if patient_name else "Not provided",
        "Patient Address": patient_address if patient_address else "Not provided",
        "Age": age,
        "Sex": sex_label,
        "BMI": bmi,
        "HbA1c": f"{hba1c}%" if hba1c is not None else "Not provided",
        "Fasting Glucose": f"{glucose} mg/dL" if glucose is not None else "Not provided",
        "Race/Ethnicity": race_label,
        "Family History": fam_label,
        "Hypertension": hyp_label,
        "Physical Activity": pa_label,
        "Smoking Status": smk_label,
    }

    # Every High outcome is ineligible, regardless of which hybrid rule fired.
    is_diabetic_lab = (hba1c is not None and hba1c >= ha.HBA1C_DIABETIC_THRESHOLD) or \
                      (glucose is not None and glucose >= ha.GLUCOSE_DIABETIC_THRESHOLD)
    has_fpg = bool(glucose is not None and math.isfinite(glucose) and glucose > 0)
    stage3_eligible = bool(tier != "High" and (not is_diabetic_lab) and has_fpg)

    return jsonify({
        "final_risk_level": tier,
        **explain_stage2(assessment, hba1c, glucose),
        "tier": tier,
        "prob_high": prob_high,
        "model1b_prediction": assessment["model1b_prediction"],
        "model1b_probability": assessment["model1b_probability"],
        "rule_triggered": rule_triggered,
        "stage3_eligible": stage3_eligible,
        "has_fasting_glucose": has_fpg,
        "reason": reasons,
        "inputs_used": assessment["inputs_used"],
        "person": person_summary
    }), 200


@app.route("/api/predict-stage3", methods=["POST"])
def predict_stage3():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid request body: expected a JSON object."}), 400

    # 1. Age (Age / age / sn256)
    raw_age = data.get("age", data.get("Age", data.get("sn256")))
    if raw_age is None or str(raw_age).strip() == "" or str(raw_age).strip().lower() == "null":
        return jsonify({"error": "Missing required field: Age"}), 400
    try:
        age_val = float(raw_age)
        if isinstance(raw_age, bool) or age_val <= 0 or not math.isfinite(age_val):
            return jsonify({"error": "Invalid value for Age: must be a positive number."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid value for Age: must be a numeric value."}), 400

    # 2. Sex (Sex / sex / sx060_r): Male / Female or 1 / 2 / 0
    raw_sex = data.get("sex", data.get("Sex", data.get("sx060_r")))
    if raw_sex is None or str(raw_sex).strip() == "" or str(raw_sex).strip().lower() == "null":
        return jsonify({"error": "Missing required field: Sex"}), 400
    s_sex = str(raw_sex).strip().lower()
    if s_sex in ("1", "1.0", "male", "m", "man"):
        sex_label = "Male"
    elif s_sex in ("2", "2.0", "0", "0.0", "female", "f", "woman"):
        sex_label = "Female"
    else:
        return jsonify({"error": "Invalid value for Sex: expected Male or Female."}), 400

    # 3. BMI (BMI / bmi)
    raw_bmi = data.get("bmi", data.get("BMI"))
    if raw_bmi is None or str(raw_bmi).strip() == "" or str(raw_bmi).strip().lower() == "null":
        return jsonify({"error": "Missing required field: BMI"}), 400
    try:
        bmi_val = float(raw_bmi)
        if isinstance(raw_bmi, bool) or bmi_val <= 0 or not math.isfinite(bmi_val):
            return jsonify({"error": "Invalid value for BMI: must be a positive number."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid value for BMI: must be a numeric value."}), 400

    # 4. Fasting Glucose (fasting_glucose_mgdl / fasting_glucose) - REQUIRED for Stage 3
    raw_glu = data.get("fasting_glucose_mgdl") if "fasting_glucose_mgdl" in data else data.get("fasting_glucose")
    if raw_glu is None or str(raw_glu).strip() in ("", "null", "none", "None"):
        return jsonify({"error": "Fasting glucose is required to generate the approximately three-year diabetes-risk projection."}), 400
    try:
        glucose_val = float(raw_glu)
        if isinstance(raw_glu, bool) or glucose_val <= 0 or not math.isfinite(glucose_val):
            return jsonify({"error": "Invalid value for fasting glucose: must be a positive number."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid value for fasting glucose: must be a numeric value."}), 400

    # 5. HbA1c (hba1c_pct / hba1c) - optional for Stage 3, checked for diagnostic range
    raw_hba1c = data.get("hba1c_pct") if "hba1c_pct" in data else data.get("hba1c")
    hba1c_val = None
    if raw_hba1c is not None and str(raw_hba1c).strip() not in ("", "null", "none", "None"):
        try:
            hba1c_val = float(raw_hba1c)
            if isinstance(raw_hba1c, bool) or hba1c_val <= 0 or not math.isfinite(hba1c_val):
                return jsonify({"error": "Invalid value for HbA1c: must be a positive number."}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid value for HbA1c: must be a numeric value."}), 400

    # 6. Diagnostic blocks: FPG >= 126 mg/dL or HbA1c >= 6.5%
    if glucose_val >= 126.0 or (hba1c_val is not None and hba1c_val >= 6.5):
        return jsonify({"error": "A diabetes-range laboratory result requires clinical confirmation; future-risk scoring is not appropriate."}), 400

    # Recompute Stage 2 from its complete inputs. Client risk labels/eligibility
    # flags cannot authorize scoring, and missing assessment inputs fail closed.
    assessment_data = dict(data)
    assessment_data.update(age=age_val, sex=sex_label, bmi=bmi_val,
                           fasting_glucose=glucose_val, hba1c=hba1c_val)
    assessment_data.pop("fasting_glucose_mgdl", None)
    assessment_response, assessment_status = assess_stage2_request(assessment_data)
    if assessment_status != 200:
        return assessment_response, assessment_status
    if not assessment_response.get_json()["stage3_eligible"]:
        return jsonify({"error": "Stage 3 is unavailable for every High Stage 2 result. Please seek clinical evaluation."}), 400

    # 7. Model evaluation via unified Rich 3-year model
    profile = {
        "age": age_val,
        "sex": sex_label.lower(),
        "bmi": bmi_val,
        "fpg_mg_dl": glucose_val
    }
    if hba1c_val is not None:
        profile["hba1c_pct"] = hba1c_val

    try:
        result = unified_s3.predict(profile)
    except S3NotApplicableError as e:
        return jsonify({"error": str(e)}), 400
    except S3OutsideEvidenceError as e:
        return jsonify({"error": str(e)}), 400
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "An unexpected error occurred during Stage 3 risk calculation."}), 500

    p3_val = float(result["risk_probability"]["3_year"])
    pct3_val = p3_val * 100
    if p3_val < 0.01:
        category = "Lower estimated risk"
        next_step = "Continue prevention and follow clinical advice on repeat screening."
    elif p3_val < 0.05:
        category = "Increased risk"
        next_step = "Discuss your risk factors and a prevention plan with a healthcare professional."
    else:
        category = "Elevated risk"
        next_step = "Arrange a clinical review to discuss prevention and follow-up."

    response_data = {
        "risk_category": category,
        "next_step": next_step,
        "risk_probability": p3_val,
        "risk_percentage": pct3_val,
        "time_horizons": {
            "3_year": {
                "probability": p3_val,
                "percentage": pct3_val,
                "evidence_scope": "research_internal_validation_only"
            }
        },
        "model_used": "unified_rich_3y",
        "reliability": result.get("reliability", "supported"),
        "warnings": result.get("warnings", []),
        "interpretation": "Estimated probability of developing diabetes within approximately three years.",
        "features_evaluated": {
            "Age": age_val,
            "Sex": sex_label,
            "BMI": bmi_val,
            "Fasting_Glucose": f"{glucose_val:.1f} mg/dL",
            "HbA1c": f"{hba1c_val:.1f}%" if hba1c_val is not None else "Not provided"
        }
    }

    patient_name = str(data.get("patient_name", data.get("Patient Name", ""))).strip()
    if patient_name:
        response_data["patient_name"] = patient_name

    return jsonify(response_data), 200


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)