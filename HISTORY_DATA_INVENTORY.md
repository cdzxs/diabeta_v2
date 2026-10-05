# Historical data inventory

Read-only audit, 2026-10-05. Original repository only; the sibling publication
copy contains no .git and none of the datasets/retired artifacts listed here.
No participant rows, credentials or sensitive field values are reproduced.

## Historical tabular data (exact paths)

XPT files below are public-use NHANES participant-level health/demographic survey
data, not locally collected DiaBeta patient records and not credentials.
SEQN is a public survey participant identifier, not a patient name. The files
are subject to NCHS statistical-use and non-identification restrictions; no
restricted-access NHANES file was identified. Public-use does not mean non-sensitive.
H/I/J correspond to 2013-14/2015-16/2017-18. Module meanings: BMX body measures;
BPQ blood pressure; DEMO demographics; DIQ diabetes questionnaire; GHB HbA1c;
GLU fasting glucose; MCQ medical conditions; PAQ activity; SMQ smoking.

| Exact historical path | Classification |
|---|---|
| `DPM/data/BMX_H.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/BMX_I.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/BMX_J.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/BPQ_H.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/BPQ_I.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/BPQ_J.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/DEMO_H.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/DEMO_I.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/DEMO_J.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/DIQ_H.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/DIQ_I.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/DIQ_J.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/GHB_H.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/GHB_I.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/GHB_J.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/GLU_H.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/GLU_I.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/GLU_J.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/MCQ_H.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/MCQ_I.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/MCQ_J.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/PAQ_H.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/PAQ_I.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/PAQ_J.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/SMQ_H.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/SMQ_I.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/SMQ_J.xpt` | NHANES public-use participant survey data; health/demographic information |
| `DPM/data/stage1_v2_nhanes_8predictors.csv` | Generated NHANES participant dataset; 16,327 rows; includes SEQN/cycle, predictors and outcome labels |
| `DPM/models/stage1_v2_test_predictions.csv` | Generated per-record labels/probabilities; 3,266 rows; no explicit participant ID/name columns |

## Two artifacts removed from the index

- `DPM/models/stage1_v2_baseline_before_integration.pkl`: generated historical
  fitted-model backup, not an active runtime dependency. Binary content cannot
  be certified free of embedded training information. No known credential use.
- `DPM/models/stage1_v2_test_predictions.csv`: generated row-level NHANES-derived
  evaluation output (y_true, y_pred and three class probabilities); no explicit
  personal identifiers. Excluded rather than treating per-record output as aggregate.

Both are now ignored and removed from the original index using `git rm --cached`;
both local files are retained. This did not remove their historical Git objects.

## Serialized and data-derived historical artifacts

These are generated models, preprocessing objects, mappings or explanations,
not raw CSV/XPT datasets. Historical SHAP arrays are per-record derived outputs.
Model binaries were not exhaustively decoded or privacy-tested. Their presence
does not prove credentials or named patient records, nor does absence of visible
identifiers prove privacy. Only current required artifacts are copied, unchanged.

- `DPM/models/diabeta_dataset1_model.pkl`
- `DPM/models/diabeta_dataset2_model.pkl`
- `DPM/models/stage1_v2_baseline_before_integration.pkl`
- `DPM/models/stage1_v2_model.pkl`
- `DPM/stage3/stage3_discrete_survival_model.json`
- `DPM/stage3/stage3_discrete_survival_model.pkl`
- `DPM/stage3/stage3_preprocessor.pkl`
- `DPM/stage3/stage3_xgboost_model.pkl`
- `DPM/stage3_baseline_backup/stage3_preprocessor.pkl`
- `DPM/stage3_baseline_backup/stage3_xgboost_model.pkl`
- `backend/model/features_nhanes_binary.pkl`
- `backend/model/model_binary_nhanes.pkl`
- `backend/model/race_map.pkl`
- `backend/model/scaler_nhanes_binary.pkl`
- `backend/model/shap_values_binary.pkl`
- `backend/model/threshold_bounds_nhanes.pkl`
- `model/features_nhanes_binary.pkl`
- `model/model_binary_nhanes.pkl`
- `model/race_map.pkl`
- `model/scaler_nhanes_binary.pkl`
- `model/shap_values_binary.pkl`
- `model/threshold_bounds_nhanes.pkl`

## Historical backup/source snapshots

- `DPM/assessment/hybrid_assessment_before_v2_model_swap.py`
- `DPM/models/stage1_v2_baseline_before_integration.pkl`
- `DPM/stage3_baseline_backup/predict_risk.py`
- `DPM/stage3_baseline_backup/stage3_preprocessor.pkl`
- `DPM/stage3_baseline_backup/stage3_threshold.txt`
- `DPM/stage3_baseline_backup/stage3_xgboost_model.pkl`
- `backend/app_before_stage2_v2_model_swap.py`
- `frontend/stage2_before_v2_model_swap.html`

## Coverage and limits

Enumerated every tree reachable from all refs: 28 commits,
118 distinct paths and 153 unique blobs. Scanned
99 decodable UTF-8 text blobs for private-key headers and common
credential patterns; 0 findings. Skipped 54 binary or
non-UTF-8 blobs for that scan. No license/notice/attribution file was found in
reachable history. No patient assessment export or restricted-access dataset
was identified by the inventory/schema review. Dataset classification follows
paths, XPT headers, schema and construction code, not re-identification.

The scan does not cover unreachable/dangling objects, reflogs, deleted refs,
external repositories, arbitrary secret encodings, image metadata exhaustively,
or model privacy attacks. Original history remains unchanged and should not be
the source of a clean-history publication. Public-use data terms are linked in
THIRD_PARTY_NOTICES.md.
