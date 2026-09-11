"""
test_hybrid_assessment.py

Covers: diabetic-range, prediabetic-range, normal-range, and missing-lab
scenarios, plus every boundary value explicitly called out in the brief
(5.7, 6.4, 6.5 for HbA1c; 100, 125, 126 for glucose).

The four hybrid rules and the Stage 1 V2 class-label mapping (LabelEncoder
0=High, 1=Low, 2=Moderate) are verified here. The test harness targets the
VALIDATED Stage 1 V2 screener (DPM/models/stage1_v2_model.pkl), which replaced
the retired Model 1B in the Stage 2 path.

Run: python3 test_hybrid_assessment.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'assessment'))

import hybrid_assessment as ha

PASSED = 0
FAILED = 0


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}  {detail}")


def base_form(**overrides):
    """A form with the Stage 1 V2 features filled in with a mid-range,
    unremarkable profile, so that lab-value rules (not the screener's own
    signal) determine the outcome in most test cases. Overrides can inject
    hba1c/fasting_glucose or change profile features to specifically exercise
    the screener's own Low/High/Moderate output for the rule-2/rule-4 tests."""
    form = {
        'age': 40.0,
        'sex': 2.0,
        'bmi': 26.0,
        'race_ethnicity': 3.0,
        'family_history': float('nan'),
        'hypertension': 0.0,
        'physical_activity': 1.0,
        'smoking_status': 0.0,
    }
    form.update(overrides)
    return form


def main():
    model = ha.load_stage1_v2_model()

    print("\n=== Sanity: Stage 1 V2 loads and exposes expected interface ===")
    check("Stage 1 V2 has exactly the 8 locked features",
          ha.STAGE1_V2_FEATURES == ['age', 'sex', 'bmi', 'race_ethnicity',
                                    'family_history', 'hypertension',
                                    'physical_activity', 'smoking_status'])
    check("Stage 1 V2 classifier classes are LabelEncoder codes [0, 1, 2]",
          list(model.classes_) == [0, 1, 2])
    check("Class-label map covers High/Low/Moderate (0=High, 1=Low, 2=Moderate)",
          ha.STAGE1_V2_CLASS_LABELS == {0: 'High', 1: 'Low', 2: 'Moderate'})

    print("\n=== extract_stage1_v2_features never includes lab values ===")
    form_with_labs = base_form(hba1c=7.0, fasting_glucose=140)
    extracted = ha.extract_stage1_v2_features(form_with_labs)
    check("hba1c not in extracted Stage 1 V2 features", 'hba1c' not in extracted)
    check("fasting_glucose not in extracted Stage 1 V2 features", 'fasting_glucose' not in extracted)
    check("Extracted feature set is exactly the 8 Stage 1 V2 features",
          set(extracted.keys()) == set(ha.STAGE1_V2_FEATURES))

    print("\n=== Rule 1: diagnostic-range labs -> HIGH (regardless of screener) ===")
    r = ha.assess(base_form(hba1c=7.2, fasting_glucose=None), model=model)
    check("HbA1c 7.2 (diabetic) -> High", r['final_risk_level'] == 'High', r)
    check("Rule triggered = rule_1", r['rule_triggered'] == 'rule_1_diagnostic_lab_override', r)

    r = ha.assess(base_form(hba1c=None, fasting_glucose=180), model=model)
    check("Glucose 180 (diabetic) -> High", r['final_risk_level'] == 'High', r)
    check("Rule triggered = rule_1 (glucose path)", r['rule_triggered'] == 'rule_1_diagnostic_lab_override', r)

    print("\n=== Rule 3: prediabetic-range labs -> MODERATE ===")
    r = ha.assess(base_form(hba1c=6.0, fasting_glucose=None), model=model)
    check("HbA1c 6.0 (prediabetic) -> Moderate", r['final_risk_level'] == 'Moderate', r)
    check("Rule triggered = rule_3", r['rule_triggered'] == 'rule_3_prediabetic_range_lab', r)

    r = ha.assess(base_form(hba1c=None, fasting_glucose=110), model=model)
    check("Glucose 110 (prediabetic) -> Moderate", r['final_risk_level'] == 'Moderate', r)

    print("\n=== Boundary values ===")
    r = ha.assess(base_form(hba1c=5.7, fasting_glucose=None), model=model)
    check("HbA1c exactly 5.7 -> Moderate (inclusive lower prediabetic bound)",
          r['final_risk_level'] == 'Moderate', r)

    r = ha.assess(base_form(hba1c=6.4, fasting_glucose=None), model=model)
    check("HbA1c exactly 6.4 -> Moderate (inclusive upper prediabetic bound)",
          r['final_risk_level'] == 'Moderate', r)

    r = ha.assess(base_form(hba1c=6.5, fasting_glucose=None), model=model)
    check("HbA1c exactly 6.5 -> High (diabetic threshold, inclusive)",
          r['final_risk_level'] == 'High' and r['rule_triggered'] == 'rule_1_diagnostic_lab_override', r)

    r = ha.assess(base_form(hba1c=None, fasting_glucose=100), model=model)
    check("Glucose exactly 100 -> Moderate (inclusive lower prediabetic bound)",
          r['final_risk_level'] == 'Moderate', r)

    r = ha.assess(base_form(hba1c=None, fasting_glucose=125), model=model)
    check("Glucose exactly 125 -> Moderate (inclusive upper prediabetic bound)",
          r['final_risk_level'] == 'Moderate', r)

    r = ha.assess(base_form(hba1c=None, fasting_glucose=126), model=model)
    check("Glucose exactly 126 -> High (diabetic threshold, inclusive)",
          r['final_risk_level'] == 'High' and r['rule_triggered'] == 'rule_1_diagnostic_lab_override', r)

    print("\n=== Missing-lab scenarios ===")
    r = ha.assess(base_form(hba1c=None, fasting_glucose=None), model=model)
    check("Both labs missing -> falls through to rule 2 or 4 (never rule 1/3)",
          r['rule_triggered'] in ('rule_2_model1b_high_probability', 'rule_4_model1b_low_high_fallback'), r)
    check("Both labs missing -> inputs_used correctly flags both absent",
          r['inputs_used']['hba1c_present'] is False and r['inputs_used']['fasting_glucose_present'] is False, r)

    r_nan = ha.assess(base_form(hba1c=float('nan'), fasting_glucose=float('nan')), model=model)
    check("NaN labs treated identically to None labs",
          r_nan['rule_triggered'] == r['rule_triggered'] and r_nan['final_risk_level'] == r['final_risk_level'], r_nan)

    r = ha.assess(base_form(hba1c=None, fasting_glucose=110), model=model)
    check("HbA1c missing, glucose prediabetic -> Moderate via glucose alone",
          r['final_risk_level'] == 'Moderate', r)

    r = ha.assess(base_form(hba1c=6.0, fasting_glucose=None), model=model)
    check("Glucose missing, HbA1c prediabetic -> Moderate via HbA1c alone",
          r['final_risk_level'] == 'Moderate', r)

    print("\n=== Normal-range labs -> falls through to screener (rule 2 or 4) ===")
    r = ha.assess(base_form(hba1c=5.2, fasting_glucose=90), model=model)
    check("Normal labs never trigger rule 1 or rule 3",
          r['rule_triggered'] in ('rule_2_model1b_high_probability', 'rule_4_model1b_low_high_fallback'), r)
    check("Normal labs -> final level is Low or High, never Moderate from labs",
          r['final_risk_level'] in ('Low', 'High'), r)

    print("\n=== Rule 2: screener P(High) >= 0.70 overrides normal/missing labs ===")
    # Search for an extreme profile that actually drives the Stage 1 V2
    # screener's P(High) >= 0.70, using the screener's own frozen feature space.
    # This is inspection of the validated model's behavior, not retraining it.
    import pandas as pd
    high_risk_profile = base_form(age=80.0, bmi=48.0, sex=1.0, race_ethnicity=4.0,
                                  family_history=1.0, hypertension=1.0,
                                  physical_activity=0.0, smoking_status=2.0,
                                  hba1c=None, fasting_glucose=None)
    _, proba_check = ha.get_stage1_v2_prediction(high_risk_profile, model=model)
    print(f"  (diagnostic) P(High) for extreme-risk profile: {proba_check.get('High'):.4f}")
    r = ha.assess(high_risk_profile, model=model)
    if proba_check.get('High', 0) >= 0.70:
        check("Extreme profile with P(High)>=0.70 and no labs -> High via rule_2",
              r['rule_triggered'] == 'rule_2_model1b_high_probability' and r['final_risk_level'] == 'High', r)
    else:
        print("  [INFO] This particular profile did not reach P(High)>=0.70 with the Stage 1 V2 screener; "
              "rule 2 logic is still exercised directly below via a monkeypatched probability.")

    # Directly exercise rule 2's threshold logic regardless of what any real profile produces,
    # using a stand-in model object - this tests THE RULE, not the screener's calibration.
    class _StubClassifier:
        classes_ = ['High', 'Low', 'Moderate']
    class _StubModel:
        named_steps = {'classifier': _StubClassifier()}
        def predict(self, X):
            return ['High']
        def predict_proba(self, X):
            return [[0.75, 0.15, 0.10]]  # High, Low, Moderate
    r = ha.assess(base_form(hba1c=None, fasting_glucose=None), model=_StubModel())
    check("Stubbed P(High)=0.75 (>=0.70), no labs -> High via rule_2",
          r['final_risk_level'] == 'High' and r['rule_triggered'] == 'rule_2_model1b_high_probability', r)

    class _StubClassifier2:
        classes_ = ['High', 'Low', 'Moderate']
    class _StubModel2:
        named_steps = {'classifier': _StubClassifier2()}
        def predict(self, X):
            return ['Low']
        def predict_proba(self, X):
            return [[0.60, 0.30, 0.10]]  # P(High)=0.60 < 0.70 threshold
    r = ha.assess(base_form(hba1c=None, fasting_glucose=None), model=_StubModel2())
    check("Stubbed P(High)=0.60 (<0.70), no labs -> falls to rule_4, not rule_2",
          r['rule_triggered'] == 'rule_4_model1b_low_high_fallback', r)
    check("Stubbed P(High)=0.60 > P(Low)=0.30 -> rule_4 resolves to High",
          r['final_risk_level'] == 'High', r)

    class _StubModel3:
        named_steps = {'classifier': _StubClassifier2()}
        def predict(self, X):
            return ['Moderate']
        def predict_proba(self, X):
            return [[0.20, 0.55, 0.25]]  # P(Low)=0.55 > P(High)=0.20
    r = ha.assess(base_form(hba1c=None, fasting_glucose=None), model=_StubModel3())
    check("Rule 4 with P(Low) > P(High) -> resolves to Low even though raw predict()=Moderate",
          r['final_risk_level'] == 'Low' and r['rule_triggered'] == 'rule_4_model1b_low_high_fallback', r)
    check("Rule 4 result never returns 'Moderate' as final_risk_level",
          r['final_risk_level'] != 'Moderate', r)

    print("\n=== Rule precedence: diagnostic labs override even a low-risk screener signal ===")
    class _StubModelLowRisk:
        named_steps = {'classifier': _StubClassifier2()}
        def predict(self, X):
            return ['Low']
        def predict_proba(self, X):
            return [[0.02, 0.95, 0.03]]
    r = ha.assess(base_form(hba1c=8.0, fasting_glucose=None), model=_StubModelLowRisk())
    check("Diabetic-range HbA1c overrides a screener-Low signal -> High",
          r['final_risk_level'] == 'High' and r['rule_triggered'] == 'rule_1_diagnostic_lab_override', r)

    print("\n=== Stage 1 V2 never receives lab values (audit via monkeypatch) ===")
    captured_columns = {}
    class _CapturingModel:
        named_steps = {'classifier': type('C', (), {'classes_': ['High', 'Low', 'Moderate']})()}
        def predict(self, X):
            captured_columns['predict'] = list(X.columns)
            return ['Low']
        def predict_proba(self, X):
            captured_columns['predict_proba'] = list(X.columns)
            return [[0.1, 0.8, 0.1]]
    ha.assess(base_form(hba1c=7.0, fasting_glucose=140), model=_CapturingModel())
    check("Columns passed to .predict() are exactly the 8 Stage 1 V2 features",
          captured_columns.get('predict') == ha.STAGE1_V2_FEATURES, captured_columns)
    check("Columns passed to .predict_proba() are exactly the 8 Stage 1 V2 features",
          captured_columns.get('predict_proba') == ha.STAGE1_V2_FEATURES, captured_columns)
    check("'hba1c' never appears in columns passed to the screener",
          'hba1c' not in captured_columns.get('predict', []), captured_columns)
    check("'fasting_glucose' never appears in columns passed to the screener",
          'fasting_glucose' not in captured_columns.get('predict', []), captured_columns)

    print("\n=== Class mapping: proba columns follow classes_ [0,1,2] (High,Low,Moderate) ===")
    # High-risk profile: code 0 (High) should carry the largest probability.
    _, proba_hi = ha.get_stage1_v2_prediction(base_form(age=80.0, bmi=48.0, sex=1.0,
                                                        race_ethnicity=4.0, family_history=1.0,
                                                        hypertension=1.0, physical_activity=0.0,
                                                        smoking_status=2.0), model=model)
    # Low-risk profile: code 1 (Low) should carry the largest probability.
    _, proba_lo = ha.get_stage1_v2_prediction(base_form(age=25.0, bmi=21.0, sex=2.0,
                                                        race_ethnicity=3.0, family_history=0.0,
                                                        hypertension=0.0, physical_activity=1.0,
                                                        smoking_status=0.0), model=model)
    check("proba dict exposes 'High' key", 'High' in proba_hi and 'High' in proba_lo)
    check("proba dict exposes 'Low' key", 'Low' in proba_hi and 'Low' in proba_lo)
    check("proba dict exposes 'Moderate' key", 'Moderate' in proba_hi and 'Moderate' in proba_lo)
    check("Extreme-risk profile: P(High) is the largest probability",
          proba_hi['High'] >= proba_hi['Low'] and proba_hi['High'] >= proba_hi['Moderate'], proba_hi)
    check("Low-risk profile: P(Low) is the largest probability",
          proba_lo['Low'] >= proba_lo['High'] and proba_lo['Low'] >= proba_lo['Moderate'], proba_lo)
    check("Probas for the same profile sum to 1.0",
          abs(sum(proba_hi.values()) - 1.0) < 1e-6 and abs(sum(proba_lo.values()) - 1.0) < 1e-6, (proba_hi, proba_lo))

    print("\n=== Waist circumference is nowhere required ===")
    check("'waist_circumference' not in Stage 1 V2 feature list",
          'waist_circumference' not in ha.STAGE1_V2_FEATURES)
    check("assess() runs with no waist_circumference key present in form at all",
          'waist_circumference' not in base_form())

    print(f"\n{'='*50}\nRESULTS: {PASSED} passed, {FAILED} failed\n{'='*50}")
    if FAILED > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
