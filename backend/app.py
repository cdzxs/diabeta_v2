import os
import sys
import math
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # allow frontend (different domain) to call the API

BASE_DIR = Path(__file__).resolve().parent
DPM_ASSESSMENT_DIR = BASE_DIR.parent / "DPM" / "assessment"
if str(DPM_ASSESSMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DPM_ASSESSMENT_DIR))

DPM_STAGE3_DIR = BASE_DIR.parent / "DPM" / "stage3"
if str(DPM_STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(DPM_STAGE3_DIR))

import hybrid_assessment as ha
import predict_risk as p_s3

# Load and memoize the Stage 3 artifacts and the Stage 1 V2 screener at startup.
# Stage 2 is the hybrid assessment layer - deterministic clinical rules over the
# validated Stage 1 V2 screener, implemented in DPM/assessment/hybrid_assessment.py
# (no separate Stage 2 model artifact). The old Model 1B (diabeta_dataset1_model.pkl)
# is retired from the active Stage 2 path; it is no longer loaded here.
stage3_prep, stage3_model, stage3_threshold = p_s3.load_artifacts()

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
        if age <= 0 or age > 130 or math.isnan(age):
            return jsonify({"error": "Invalid value for age: must be a positive number up to 130."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid value for age: must be a numeric value."}), 400

    # bmi
    if "bmi" not in data or data["bmi"] is None or str(data["bmi"]).strip() == "":
        return jsonify({"error": "Missing required field: bmi"}), 400
    try:
        bmi = float(data["bmi"])
        if bmi <= 0 or bmi > 150 or math.isnan(bmi):
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
        race_code = int(float(data["race_ethnicity"]))
    except (ValueError, TypeError):
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
    optional HbA1c and fasting glucose (either, both, or neither may be
    supplied), and runs the existing hybrid assessment engine
    (DPM/assessment/hybrid_assessment.py) on top of the validated Stage 1 V2
    screener.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid request body: expected a JSON object."}), 400

    # 1. Validate & parse the five Stage 1 screener inputs
    # age
    if "age" not in data or data["age"] is None or str(data["age"]).strip() == "":
        return jsonify({"error": "Missing required field: age"}), 400
    try:
        age = float(data["age"])
        if age <= 0 or age > 130 or math.isnan(age):
            return jsonify({"error": "Invalid value for age: must be a positive number up to 130."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid value for age: must be a numeric value."}), 400

    # bmi
    if "bmi" not in data or data["bmi"] is None or str(data["bmi"]).strip() == "":
        return jsonify({"error": "Missing required field: bmi"}), 400
    try:
        bmi = float(data["bmi"])
        if bmi <= 0 or bmi > 150 or math.isnan(bmi):
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
        race_code = int(float(data["race_ethnicity"]))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid value for race_ethnicity: must be an integer code."}), 400
    race_label = race_map.get(race_code, race_map.get(str(race_code), f"Code {race_code}"))

    # family_history (1 -> 1.0, 0 -> NaN, matching Stage 1 semantics)
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
            if math.isnan(hba1c) or hba1c < 0:
                return jsonify({"error": "Invalid value for hba1c: must be a positive number."}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid value for hba1c: must be a numeric value."}), 400

    # Fasting glucose (accepts fasting_glucose_mgdl or fasting_glucose)
    glucose = None
    raw_glucose = data.get("fasting_glucose_mgdl") if "fasting_glucose_mgdl" in data else data.get("fasting_glucose")
    if raw_glucose is not None and str(raw_glucose).strip() != "" and str(raw_glucose).strip().lower() != "null":
        try:
            glucose = float(raw_glucose)
            if math.isnan(glucose) or glucose < 0:
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
            reasons.append("Laboratory values were within normal reference ranges; classification determined by the ML screener.")
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

    # Stage 3 eligibility: time-based future diabetes risk is only meaningful
    # when the Stage 2 assessment did NOT already conclude HIGH risk. ANY final
    # HIGH result - whether from Rule 1 (diabetes-range labs), Rule 2 (screener
    # P(High) >= 0.70), or Rule 4 (screener Low/High fallback) - blocks Stage 3.
    # Only LOW and MODERATE Stage 2 outcomes may proceed.
    stage3_eligible = tier in ("Low", "Moderate")

    return jsonify({
        "final_risk_level": tier,
        "tier": tier,
        "prob_high": prob_high,
        "model1b_prediction": assessment["model1b_prediction"],
        "model1b_probability": assessment["model1b_probability"],
        "rule_triggered": rule_triggered,
        "stage3_eligible": stage3_eligible,
        "reason": reasons,
        "inputs_used": assessment["inputs_used"],
        "person": person_summary
    }), 200


@app.route("/api/predict-stage3", methods=["POST"])
def predict_stage3():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid request body: expected a JSON object."}), 400

    # Accept raw 13 Stage 3 features or mapped patient profile from Stage 1/Stage 2
    # 1. Age (sn256 or Age / age)
    raw_age = data.get("sn256", data.get("Age", data.get("age")))
    if raw_age is None or str(raw_age).strip() == "" or str(raw_age).strip().lower() == "null":
        return jsonify({"error": "Missing required field: Age (sn256)"}), 400
    try:
        age_val = float(raw_age)
        if age_val <= 0 or age_val > 130 or math.isnan(age_val):
            return jsonify({"error": "Invalid value for Age: must be a positive number up to 130."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid value for Age: must be a numeric value."}), 400

    # 2. Sex (sx060_r or Sex / sex): SHARE coding 1 = Male, 2 = Female
    raw_sex = data.get("sx060_r", data.get("Sex", data.get("sex")))
    if raw_sex is None or str(raw_sex).strip() == "" or str(raw_sex).strip().lower() == "null":
        return jsonify({"error": "Missing required field: Sex (sx060_r)"}), 400
    s_sex = str(raw_sex).strip().lower()
    if s_sex in ("1", "1.0", "male", "m"):
        sex_val = 1
    elif s_sex in ("2", "2.0", "0", "0.0", "female", "f"):
        sex_val = 2
    else:
        return jsonify({"error": "Invalid value for Sex: expected Male or Female."}), 400

    # ---- Stage 3 eligibility safeguard (defense-in-depth) ----
    # If the Stage 2 assessment recorded a diabetes-range laboratory finding
    # (Rule 1: HbA1c >= 6.5% OR fasting glucose >= 126 mg/dL), future
    # diabetes-development risk is NOT applicable. The protected Stage 3
    # workflow passes the Stage 2 laboratory values here so the server
    # independently enforces eligibility - it never trusts a client-supplied
    # eligibility flag alone. Requests without laboratory values keep the
    # existing backward-compatible behavior. The check runs before any Stage 3
    # risk calculation.
    raw_hba1c3 = data.get("hba1c")
    s3_hba1c = None
    if raw_hba1c3 is not None and str(raw_hba1c3).strip() not in ("", "null", "None"):
        try:
            s3_hba1c = float(raw_hba1c3)
            if math.isnan(s3_hba1c) or s3_hba1c < 0:
                return jsonify({"error": "Invalid value for hba1c: must be a positive number."}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid value for hba1c: must be a numeric value."}), 400

    raw_glu3 = data.get("fasting_glucose_mgdl") if "fasting_glucose_mgdl" in data else data.get("fasting_glucose")
    s3_glucose = None
    if raw_glu3 is not None and str(raw_glu3).strip() not in ("", "null", "None"):
        try:
            s3_glucose = float(raw_glu3)
            if math.isnan(s3_glucose) or s3_glucose < 0:
                return jsonify({"error": "Invalid value for fasting glucose: must be a positive number."}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid value for fasting glucose: must be a numeric value."}), 400

    if (s3_hba1c is not None and s3_hba1c >= ha.HBA1C_DIABETIC_THRESHOLD) or \
       (s3_glucose is not None and s3_glucose >= ha.GLUCOSE_DIABETIC_THRESHOLD):
        return jsonify({"error": "Diabetes-range result detected. Time-based diabetes development risk is not applicable. Please seek appropriate clinical evaluation."}), 400

    # ---- Stage 3 eligibility: ANY Stage 2 HIGH result blocks (defense-in-depth) ----
    # A final Stage 2 outcome of HIGH - from Rule 1, Rule 2, or Rule 4 - means
    # time-based future diabetes risk is NOT applicable. The protected Stage 2
    # workflow passes its final risk tier here (stage2_tier) so the server can
    # independently enforce this, on top of the diabetes-range laboratory check
    # above. Requests without a Stage 2 tier (e.g. a direct Stage 1 or
    # standalone Stage 3 entry) keep the existing backward-compatible behavior.
    raw_s2_tier = data.get("stage2_tier", data.get("final_risk_level", data.get("tier")))
    if raw_s2_tier is not None and str(raw_s2_tier).strip().lower() == "high":
        return jsonify({"error": "High-risk Stage 2 result detected. Time-based diabetes development risk is not applicable. Please seek appropriate clinical evaluation."}), 400

    # Helper for condition indicators (1 = Yes, 0 = No)
    def _parse_flag(field_primary: str, field_secondary: str, default: int = 0) -> int:
        val = data.get(field_primary, data.get(field_secondary))
        if val is None or str(val).strip() == "" or str(val).strip().lower() == "null":
            return default
        s = str(val).strip().lower()
        if s in ("1", "1.0", "true", "yes"):
            return 1
        return 0

    hyp = _parse_flag("sz101", "Hypertension")
    heart = _parse_flag("sz103", "Heart_Disease")
    stroke = _parse_flag("sz107", "Stroke")
    chol = _parse_flag("sz124", "High_Cholesterol")

    # Additional co-morbidities (default to 0 if not provided)
    diabetes_hist = _parse_flag("sz105", "Diabetes_History", default=0)
    lung_disease = _parse_flag("sz106", "Lung_Disease", default=0)
    heart_failure = _parse_flag("sz104", "Heart_Failure", default=0)
    arthritis = _parse_flag("sz108", "Arthritis", default=0)
    cancer = _parse_flag("sz122", "Cancer", default=0)
    kidney = _parse_flag("sz123", "Kidney_Disease", default=0)

    # Condition count (sz080)
    if "sz080" in data and data["sz080"] is not None and str(data["sz080"]).strip() != "" and str(data["sz080"]).strip().lower() != "null":
        try:
            cond_count = int(float(data["sz080"]))
        except (ValueError, TypeError):
            cond_count = hyp + heart + stroke + chol + diabetes_hist + lung_disease + heart_failure + arthritis + cancer + kidney
    else:
        cond_count = hyp + heart + stroke + chol + diabetes_hist + lung_disease + heart_failure + arthritis + cancer + kidney

    # Assemble exact 13 required features for Stage 3
    s3_patient = {
        'sn256': age_val,
        'sx060_r': sex_val,
        'sz101': hyp,
        'sz105': diabetes_hist,
        'sz106': lung_disease,
        'sz103': heart,
        'sz104': heart_failure,
        'sz107': stroke,
        'sz108': arthritis,
        'sz122': cancer,
        'sz123': kidney,
        'sz124': chol,
        'sz080': cond_count,
    }

    try:
        raw_result = p_s3.predict_stage3_risk(s3_patient)
    except Exception as e:
        return jsonify({"error": f"Stage 3 prediction error: {str(e)}"}), 500

    risk_prob = raw_result["risk_probability"]
    risk_pct = raw_result["risk_percentage"]

    # Genuine discrete-time survival horizons: strictly 1-Year and 2-Year risk
    horizons = raw_result.get("time_horizons", {})
    r1 = horizons.get("1_year", {}).get("probability", raw_result.get("risk_probability_1yr", 0.0))
    p1_pct = horizons.get("1_year", {}).get("percentage", raw_result.get("risk_percentage_1yr", 0.0))
    r2 = horizons.get("2_year", {}).get("probability", raw_result.get("risk_probability_2yr", risk_prob))
    p2_pct = horizons.get("2_year", {}).get("percentage", raw_result.get("risk_percentage_2yr", risk_pct))

    response_data = {
        "risk_probability": risk_prob,
        "risk_percentage": risk_pct,
        "time_horizons": {
            "1_year": {
                "probability": round(float(r1), 4),
                "percentage": round(float(p1_pct), 2)
            },
            "2_year": {
                "probability": round(float(r2), 4),
                "percentage": round(float(p2_pct), 2)
            }
        },
        "features_evaluated": {
            "Age": age_val,
            "Sex": "Male" if sex_val == 1 else "Female",
            "Hypertension": "Yes" if hyp == 1 else "No",
            "High_Cholesterol": "Yes" if chol == 1 else "No",
            "Heart_Disease": "Yes" if heart == 1 else "No",
            "Stroke": "Yes" if stroke == 1 else "No",
            "Chronic_Condition_Count": cond_count
        }
    }

    patient_name = str(data.get("patient_name", data.get("Patient Name", ""))).strip()
    if patient_name:
        response_data["patient_name"] = patient_name

    return jsonify(response_data), 200


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)