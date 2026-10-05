"""Run from project root: python -B -m unittest discover -s backend -p test_stage3_integration.py.

Uses real serialized models for loading/inference; overrides only the Stage 1
prediction in rule tests to exercise each deterministic hybrid rule reliably.
"""
import unittest
import math
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import app as backend


class Stage3IntegrationTests(unittest.TestCase):
    def test_separate_lab_profile_and_final_explanations(self):
        for hba1c, glucose, probabilities, tier, rule, lab_phrase in [
            (6.5, 90, {'High': .1, 'Low': .8, 'Moderate': .1}, 'High', 'rule_1', 'Diabetes-range'),
            (5.2, 90, {'High': .70, 'Low': .2, 'Moderate': .1}, 'High', 'rule_2', 'Below'),
            (5.9, None, {'High': .1, 'Low': .8, 'Moderate': .1}, 'Moderate', 'rule_3', 'Prediabetes-range'),
            (None, 110, {'High': .1, 'Low': .8, 'Moderate': .1}, 'Moderate', 'rule_3', 'Not provided'),
            (5.2, 90, {'High': .3, 'Low': .2, 'Moderate': .5}, 'High', 'rule_4', 'Below'),
            (5.2, 90, {'High': .1, 'Low': .8, 'Moderate': .1}, 'Low', 'rule_4', 'Below'),
            (6.45, 125.5, {'High': .1, 'Low': .8, 'Moderate': .1}, 'Low', 'rule_4', 'Between'),
        ]:
            with self.subTest(rule=rule, tier=tier), patch.object(backend.ha, 'get_stage1_v2_prediction',
                    return_value=('Moderate', probabilities)):
                data = self.client.post('/api/predict-stage2', json=dict(self.profile,
                    hba1c=hba1c, fasting_glucose=glucose)).get_json()
            self.assertEqual(data['final_recommendation']['tier'], tier)
            self.assertTrue(data['final_recommendation']['rule'].startswith(rule))
            self.assertIn(lab_phrase, data['lab_assessment'][0]['finding'])
            self.assertEqual(data['profile_screening']['high_probability'], probabilities['High'])
            self.assertTrue(data['final_recommendation']['next_step'])
            self.assertEqual(data['stage3_eligible'], tier != 'High' and glucose is not None)
            if glucose is None:
                self.assertIn('Not provided', data['lab_assessment'][1]['finding'])
            script = Path(__file__).resolve().parents[1] / 'frontend/tests/stage2.test.cjs'
            run = subprocess.run(['node', str(script)], capture_output=True, text=True,
                                 env=dict(os.environ, STAGE2_EXPLANATION=json.dumps(data)))
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_screenshot_profiles_and_hba1c_is_not_a_predictor(self):
        results = []
        profile_percentages = []
        for glucose, hba1c, percentage, reliability in [
            (112, 6.4, 2.55, 'supported'), (122, None, 4.25, 'caution'),
            (113, 6.1, 2.55, 'supported'),
        ]:
            payload = dict(self.profile, age=25, sex=1, bmi=21,
                           fasting_glucose=glucose, hba1c=hba1c)
            s2 = self.client.post('/api/predict-stage2', json=payload).get_json()
            self.assertEqual(s2['tier'], 'Moderate')
            profile_percentages.append(s2['prob_high'])
            response = self.client.post('/api/predict-stage3', json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(round(data['risk_percentage'], 2), percentage)
            self.assertEqual(data['risk_category'], 'Increased risk')
            self.assertEqual(data['reliability'], reliability)
            self.assertEqual(bool(data['warnings']), reliability == 'caution')
            without_hba1c = self.client.post('/api/predict-stage3', json=dict(payload, hba1c=None)).get_json()
            self.assertEqual(data['risk_probability'], without_hba1c['risk_probability'])
            results.append(data)
        self.assertEqual(results[0]['risk_probability'], results[2]['risk_probability'])
        self.assertEqual(len(set(profile_percentages)), 1)
        self.assertGreater(results[1]['risk_probability'], results[0]['risk_probability'])
        frontend = Path(__file__).resolve().parents[1] / 'frontend/tests/stage3.test.cjs'
        run = subprocess.run(['node', str(frontend)], capture_output=True, text=True,
                             env=dict(os.environ, STAGE3_REAL_RESULTS=json.dumps(results)))
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_exact_59_96_page_flow_and_missing_glucose_notice(self):
        moderate = dict(self.profile, hba1c=5.9, fasting_glucose=96)
        cases = dict(missing=dict(moderate, fasting_glucose=None), moderate=moderate,
                     high=dict(moderate, hba1c=7))
        responses = {}
        for name, payload in cases.items():
            response = self.client.post('/api/predict-stage2', json=payload)
            self.assertEqual(response.status_code, 200)
            responses[name] = response.get_json()
        self.assertEqual(responses['missing']['tier'], 'Moderate')
        self.assertFalse(responses['missing']['stage3_eligible'])
        self.assertIsNone(responses['missing']['inputs_used']['fasting_glucose'])
        self.assertEqual(responses['moderate']['tier'], 'Moderate')
        self.assertTrue(responses['moderate']['stage3_eligible'])
        self.assertEqual(responses['moderate']['inputs_used']['fasting_glucose'], 96)
        script = Path(__file__).resolve().parents[1] / 'frontend/tests/stage2.test.cjs'
        run = subprocess.run(['node', str(script)], capture_output=True, text=True,
                             env=dict(os.environ, STAGE2_FLOW=json.dumps(responses)))
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        traces = json.loads(next(line.removeprefix('FLOW_RESULT:') for line in run.stdout.splitlines()
                                 if line.startswith('FLOW_RESULT:')))
        for payload, status in zip(traces, [400, 200, 400]):
            result = self.client.post('/api/predict-stage3', json=payload)
            self.assertEqual(result.status_code, status, result.get_json())

    def trace_frontend(self, payload, response):
        frontend = Path(__file__).resolve().parents[1] / 'frontend' / 'tests' / 'stage3.test.cjs'
        run = subprocess.run(['node', str(frontend)], capture_output=True, text=True,
                             env=dict(os.environ, STAGE2_TRACE=json.dumps(
                                 dict(payload=payload, resData=response))))
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        return json.loads(next(line.removeprefix('TRACE_RESULT:')
                               for line in run.stdout.splitlines() if line.startswith('TRACE_RESULT:')))

    def test_moderate_storage_payload_and_reassessment_match(self):
        payload = {k: str(v) for k, v in dict(self.profile, fasting_glucose=100, hba1c=6).items()}
        with patch.object(backend.ha, 'assess', wraps=backend.ha.assess) as assessment:
            s2 = self.client.post('/api/predict-stage2', json=payload).get_json()
            self.assertEqual(s2['final_risk_level'], 'Moderate')
            self.assertTrue(s2['stage3_eligible'])
            trace = self.trace_frontend(payload, s2)
            self.assertEqual(trace['record']['raw_inputs'], payload)
            self.assertEqual(trace['payload'], payload)
            result = self.client.post('/api/predict-stage3', json=trace['payload'])
            self.assertEqual(result.status_code, 200, result.get_json())
            self.assertEqual(assessment.call_args_list[0].args[0], assessment.call_args_list[1].args[0])

    def test_high_storage_blocks_and_forged_moderate_cannot_bypass(self):
        payload = dict(self.profile, hba1c=7)
        s2 = self.client.post('/api/predict-stage2', json=payload).get_json()
        self.assertEqual(s2['final_risk_level'], 'High')
        trace = self.trace_frontend(payload, s2)
        self.assertTrue(trace['blocked'])
        with patch.object(backend.unified_s3, 'predict') as scoring:
            result = self.client.post('/api/predict-stage3', json=dict(
                payload, tier='Moderate', final_risk_level='Moderate', stage3_eligible=True))
            self.assertEqual(result.status_code, 400)
            scoring.assert_not_called()

    def test_stale_hba1c_can_change_moderate_to_high(self):
        # Reproduce the old mixed-input failure: the stored request has HbA1c 6,
        # but its display summary has 5.2, removing the Moderate lab rule.
        with patch.object(backend.ha, 'get_stage1_v2_prediction',
                          return_value=('High', {'High': .6, 'Low': .3, 'Moderate': .1})):
            payload = dict(self.profile, hba1c=6)
            s2 = self.client.post('/api/predict-stage2', json=payload).get_json()
            self.assertEqual(s2['final_risk_level'], 'Moderate')
            old_payload = dict(payload, hba1c=5.2)
            self.assertEqual(self.client.post('/api/predict-stage3', json=old_payload).status_code, 400)
            self.assertEqual(self.client.post('/api/predict-stage3', json=payload).status_code, 200)

    def setUp(self):
        self.client = backend.app.test_client()
        self.profile = dict(age=45, sex=2, bmi=27, race_ethnicity=3,
                            family_history=0, hypertension=0, physical_activity=1,
                            smoking_status=0, fasting_glucose=90, hba1c=5.2)

    def test_raw_probability_boundaries(self):
        for probability, category in [
            (math.nextafter(.01, 0), 'Lower estimated risk'),
            (.01, 'Increased risk'),
            (math.nextafter(.05, 0), 'Increased risk'),
            (.05, 'Elevated risk'),
        ]:
            with self.subTest(probability=probability), patch.object(
                    backend.ha, 'get_stage1_v2_prediction',
                    return_value=('Low', {'High': .1, 'Low': .8, 'Moderate': .1})), patch.object(
                    backend.unified_s3, 'predict', return_value={
                        'risk_probability': {'3_year': probability},
                        'risk_percent': {'3_year': round(probability * 100, 2)},
                        'warnings': ['Evidence warning'], 'reliability': 'caution'}):
                result = self.client.post('/api/predict-stage3', json=self.profile)
            self.assertEqual(result.status_code, 200)
            data = result.get_json()
            self.assertEqual(data['risk_category'], category)
            self.assertEqual(data['risk_probability'], probability)
            self.assertEqual(data['time_horizons']['3_year']['probability'], probability)
            self.assertEqual(data['risk_percentage'], probability * 100)
            self.assertEqual(data['warnings'], ['Evidence warning'])
            self.assertEqual(data['reliability'], 'caution')
            self.assertTrue(data['next_step'])

    def test_eligible_real_model(self):
        with patch.object(backend.ha, 'get_stage1_v2_prediction',
                          return_value=('Low', {'High': .1, 'Low': .8, 'Moderate': .1})):
            result = self.client.post('/api/predict-stage3', json=self.profile)
        self.assertEqual(result.status_code, 200, result.get_json())
        data = result.get_json()
        self.assertEqual(data['model_used'], 'unified_rich_3y')
        self.assertEqual(list(data['time_horizons']), ['3_year'])
        self.assertNotIn('validated', data['time_horizons']['3_year'])

    def test_real_models_all_categories_and_frontend(self):
        results = []
        for age, bmi, glucose, category, advice in [
            (30, 22, 85, 'Lower estimated risk', 'repeat screening'),
            (45, 27, 100, 'Increased risk', 'prevention plan'),
            (60, 30, 115, 'Elevated risk', 'clinical review'),
            (60, 30, 120, 'Elevated risk', 'clinical review'),
        ]:
            # No patches: both the Stage 2 eligibility model and Stage 3 model run.
            profile = dict(self.profile, age=age, bmi=bmi, fasting_glucose=glucose)
            with self.subTest(profile=profile):
                response = self.client.post('/api/predict-stage3', json=profile)
                self.assertEqual(response.status_code, 200, response.get_json())
                data = response.get_json()
                self.assertEqual(data['risk_category'], category)
                self.assertIn(advice, data['next_step'])
                raw = backend.unified_s3.predict(dict(age=age, sex='female', bmi=bmi,
                                                     fpg_mg_dl=glucose, hba1c_pct=5.2))
                self.assertEqual(data['risk_probability'], raw['risk_probability']['3_year'])
                self.assertEqual(data['risk_percentage'], data['risk_probability'] * 100)
                self.assertEqual(data['warnings'], raw['warnings'])
                if glucose == 120:
                    self.assertEqual(data['reliability'], 'caution')
                    self.assertTrue(data['warnings'])
                results.append(data)
        frontend = Path(__file__).resolve().parents[1] / 'frontend' / 'tests' / 'stage3.test.cjs'
        run = subprocess.run(['node', str(frontend)], capture_output=True, text=True,
                             env=dict(os.environ, STAGE3_REAL_RESULTS=json.dumps(results)))
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn('real model output displays and restores: Lower estimated risk', run.stdout)

    def test_every_high_rule_cannot_be_bypassed_by_low_label(self):
        for rule, probabilities, labs in [
            ('rule_1_diagnostic_lab_override', {'High': .1, 'Low': .8, 'Moderate': .1}, {'hba1c': 6.5}),
            ('rule_2_model1b_high_probability', {'High': .75, 'Low': .15, 'Moderate': .1}, {}),
            ('rule_4_model1b_low_high_fallback', {'High': .6, 'Low': .3, 'Moderate': .1}, {}),
        ]:
            with self.subTest(rule=rule), patch.object(backend.ha, 'get_stage1_v2_prediction',
                    return_value=('High', probabilities)), patch.object(backend.unified_s3, 'predict') as scoring:
                payload = dict(self.profile, **labs, final_risk_level='Low', stage3_eligible=True)
                s2 = self.client.post('/api/predict-stage2', json=payload).get_json()
                self.assertEqual(s2['rule_triggered'], rule)
                self.assertFalse(s2['stage3_eligible'])
                self.assertEqual(self.client.post('/api/predict-stage3', json=payload).status_code, 400)
                scoring.assert_not_called()

    def test_missing_inputs_and_lab_blocks(self):
        cases = [dict(self.profile, fasting_glucose=value) for value in (None, '', ' ', 0, 'bad', 126, 140)]
        cases += [dict(self.profile, hba1c=value) for value in (6.5, 7)]
        for field in ('fasting_glucose', 'race_ethnicity', 'family_history', 'hypertension', 'physical_activity', 'smoking_status'):
            payload = dict(self.profile)
            del payload[field]
            cases.append(payload)
        with patch.object(backend.unified_s3, 'predict') as scoring:
            for payload in cases:
                with self.subTest(payload=payload):
                    self.assertEqual(self.client.post('/api/predict-stage3', json=payload).status_code, 400)
            scoring.assert_not_called()


if __name__ == '__main__':
    unittest.main()
