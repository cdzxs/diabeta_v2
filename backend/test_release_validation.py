"""Release regressions: validation, mapping, and deterministic real inference."""
import unittest
from unittest.mock import patch
import app as backend


class ReleaseValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = backend.app.test_client()
        self.profile = dict(age=45, sex=2, bmi=27, race_ethnicity=3,
                            family_history=0, hypertension=0, physical_activity=1,
                            smoking_status=0, fasting_glucose=100, hba1c=5.9)

    def test_invalid_numeric_inputs_fail_without_server_error(self):
        for endpoint in ('predict', 'predict-stage2', 'predict-stage3'):
            for field in ('age', 'bmi', 'race_ethnicity'):
                values = [True, False, 'NaN', 'Infinity', '-Infinity', [], {}, 0, -1]
                if field == 'race_ethnicity':
                    values += [1.5, 5, 99]
                for value in values:
                    with self.subTest(endpoint=endpoint, field=field, value=value):
                        response = self.client.post('/api/' + endpoint,
                            json=dict(self.profile, **{field: value}))
                        self.assertEqual(response.status_code, 400)

    def test_stage2_and_stage3_reject_invalid_labs(self):
        for endpoint in ('predict-stage2', 'predict-stage3'):
            for field in ('hba1c', 'fasting_glucose'):
                for value in (0, -1, True, False, 'NaN', 'Infinity', '-Infinity', [], {}):
                    with self.subTest(endpoint=endpoint, field=field, value=value):
                        self.assertEqual(self.client.post('/api/' + endpoint,
                            json=dict(self.profile, **{field: value})).status_code, 400)

    def test_required_profiles_and_json_body(self):
        for endpoint in ('predict', 'predict-stage2', 'predict-stage3'):
            for field in backend.STAGE1_FEATURES:
                profile = dict(self.profile)
                del profile[field]
                self.assertEqual(self.client.post('/api/' + endpoint, json=profile).status_code, 400)
            for body in ([], 'text', 42):
                self.assertEqual(self.client.post('/api/' + endpoint, json=body).status_code, 400)

    def test_sex_mapping_predictors_and_units(self):
        for sex, nhanes, male in [(0, 2., 0), (2, 2., 0), ('female', 2., 0), (1, 1., 1), ('male', 1., 1)]:
            profile = dict(self.profile, sex=sex)
            with patch.object(backend.stage1_model, 'predict', wraps=backend.stage1_model.predict) as predict:
                self.assertEqual(self.client.post('/api/predict', json=profile).status_code, 200)
                row = predict.call_args.args[0]
                self.assertEqual(list(row.columns), backend.STAGE1_FEATURES)
                self.assertEqual(row.iloc[0]['sex'], nhanes)
            with patch.object(backend.ha, 'assess', wraps=backend.ha.assess) as assess:
                self.assertEqual(self.client.post('/api/predict-stage2', json=profile).status_code, 200)
                self.assertEqual(assess.call_args.args[0]['sex'], nhanes)
            model = backend._unified_s3_model
            with patch.object(backend.unified_s3.joblib, 'load', return_value=model), patch.object(
                    model, 'predict_proba', wraps=model.predict_proba) as predict:
                self.assertEqual(self.client.post('/api/predict-stage3', json=profile).status_code, 200)
                row = predict.call_args.args[0][0]
                self.assertEqual(list(row[:3]), [45., male, 27.])
                self.assertAlmostEqual(row[3], 100 / 18.0182)

    def test_real_flow_direct_entry_and_repeated_inputs(self):
        self.assertEqual(self.client.post('/api/predict', json=self.profile).status_code, 200)
        s2 = self.client.post('/api/predict-stage2', json=self.profile)
        self.assertEqual(s2.status_code, 200)
        self.assertTrue(s2.get_json()['stage3_eligible'])
        predictions = [self.client.post('/api/predict-stage3', json=self.profile).get_json()
                       for _ in range(3)]
        self.assertEqual(predictions[0], predictions[1])
        self.assertEqual(predictions[1], predictions[2])
        direct = backend.app.test_client().post('/api/predict-stage2', json=self.profile)
        self.assertEqual(s2.get_json(), direct.get_json())


if __name__ == '__main__':
    unittest.main()
