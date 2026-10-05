"""Read-only reconstructed held-out policy replay. Run from project root with Python -B.
Writes no model or source-data artifacts; prints aggregate JSON only.
"""
import json
from pathlib import Path
import numpy as np
import joblib
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
import app

root = Path(__file__).resolve().parents[1] / 'DPM'
df = pd.read_csv(root / 'data/stage1_v2_nhanes_8predictors.csv')
_, test = train_test_split(df, test_size=.2, stratify=df.diabetes_risk, random_state=42)
saved = pd.read_csv(root / 'models/stage1_v2_test_predictions.csv')
proba = app.stage1_model.predict_proba(test[app.STAGE1_FEATURES])
assert list(test.diabetes_risk) == list(saved.y_true), 'Held-out row order not verified'
saved_matches_current = np.allclose(proba, saved[['p_High', 'p_Low', 'p_Moderate']], atol=1e-12)
baseline = joblib.load(root / 'models/stage1_v2_baseline_before_integration.pkl')
assert np.allclose(baseline.predict_proba(test[app.STAGE1_FEATURES]),
                   saved[['p_High', 'p_Low', 'p_Moderate']], atol=1e-12), 'Baseline/split mismatch'
labs = []
for cycle, suffix in [('2013-2014', 'H'), ('2015-2016', 'I'), ('2017-2018', 'J')]:
    a = pd.read_sas(root / f'data/GHB_{suffix}.xpt', format='xport')[['SEQN','LBXGH']]
    g = pd.read_sas(root / f'data/GLU_{suffix}.xpt', format='xport')[['SEQN','LBXGLU']]
    joined = a.merge(g, on='SEQN', how='outer', validate='one_to_one')
    joined['cycle'] = cycle
    labs.append(joined)
test = test.assign(p_high=proba[:,0], p_low=proba[:,1]).merge(
    pd.concat(labs), on=['SEQN','cycle'], how='left', validate='one_to_one')
h, g = test.LBXGH, test.LBXGLU
present = h.notna() | g.notna()
diagnostic = (h >= 6.5) | (g >= 126)
r2 = ~diagnostic & (test.p_high >= .70) & present
prediabetes = h.between(5.7,6.4) | g.between(100,125)
r4 = ~diagnostic & ~r2 & ~prediabetes & (test.p_high >= test.p_low) & present
both_normal = h.gt(0) & h.lt(5.7) & g.gt(0) & g.lt(100)
available_normal = present & (h.isna() | (h.gt(0) & h.lt(5.7))) & (g.isna() | (g.gt(0) & g.lt(100)))
evidence = test.age.between(20,99) & test.bmi.between(15,52.7) & g.ge(50.45) & g.lt(126) & ~diagnostic
complete = test[app.STAGE1_FEATURES].notna().all(axis=1)
out = dict(held_out_rows=len(test), split_verified_against_baseline=True,
           saved_predictions_match_active_model=bool(saved_matches_current),
           active_model_sha256=hashlib.sha256((root / 'models/stage1_v2_model.pkl').read_bytes()).hexdigest(),
           both_normal=int(both_normal.sum()), available_normal=int(available_normal.sum()),
           otherwise_stage3_evidence_population=int((present & evidence).sum()),
           no_labs=int((~present).sum()), source='NHANES held-out sample; unweighted policy replay, not app users')
for name, mask in [('rule2', r2), ('rule4', r4)]:
    out[name] = dict(final_high=int(mask.sum()), both_normal_high=int((mask & both_normal).sum()),
        available_normal_high=int((mask & available_normal).sum()),
        otherwise_stage3_evidence_eligible=int((mask & evidence).sum()),
        complete_profile_and_stage3_evidence=int((mask & evidence & complete).sum()),
        known_diabetes_and_stage3_evidence=int((mask & evidence & test.diq010.eq(1)).sum()),
        both_normal_known_diabetes=int((mask & both_normal & test.diq010.eq(1)).sum()))
print(json.dumps(out, indent=2))
