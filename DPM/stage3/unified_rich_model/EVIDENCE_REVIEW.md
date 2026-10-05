# Stage 3 evidence review (2026-10-05)

This is a research prototype, not a diagnosis or a clinically validated Nigerian
risk calculator. No model was retrained, no threshold changed, and no training
predictions were presented as held-out evaluation in this audit.

## Artifact and inference

The API imports this directory's inference.py and loads
output/unified_rich_3y_model.joblib at startup and at inference. SHA-256:
`110600c8ec947a3c5003a1680332af1ae2dec4b7ce6ac4e0c66ee416284d9d2e`.
It matches the local research candidate. The serialized object is a four-feature
HistGradientBoostingClassifier with classes [0, 1], monotonic constraints
[1, 0, 1, 1], max_iter=180, max_leaf_nodes=12, min_samples_leaf=50,
learning_rate=0.05, l2_regularization=5, random_state=20260920.

The exact numeric row is [age in years, male 1/female 0, BMI in kg/m²,
fasting glucose in mmol/L]. The application accepts fasting glucose in mg/dL
and divides by 18.0182. Stage 1/2 instead use NHANES male 1/female 2; the
API's conversion through a sex label was tested for female 0/2 and male 1.
There is no external Stage 3 scaler, imputer, or calibrator to ship. HbA1c
affects eligibility only and is not one of the four predictors.

## Source, eligibility, and outcome

Source: Chen et al.'s [Rich Healthcare dataset on Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.ft8750v).
Its description identifies baseline diabetes-free Chinese health-check adults,
with diabetes detected by fasting glucose >=7 mmol/L and/or self-report at
follow-up visits. These are visit-detected outcomes, not exact onset dates.
See THIRD_PARTY_NOTICES.md for attribution and terms.

The supplied training source filters baseline FPG to 2.8 <= FPG < 7 mmol/L,
rejects missing core inputs/outcomes, and reports 211,783 eligible rows.
The fixed-horizon label is event=1 AND follow-up <=3 years. Evaluable rows are
those cases OR follow-up >3 years: 105,740 rows and 1,763 cases. Event-free
follow-up exactly at 3 years is excluded by the actual `>` expression.
This detail is recorded, not changed. Earlier censored participants contribute
to the censoring distribution but not binary-model fitting.

The source workbook and raw baseline exclusion checks are not present in this
workspace. Reported source SHA-256 is
`ffd341a1a91ac65585b9fba70847a2681f5671a6035d9cfbf92e50795d74fbf6`;
this audit did not independently verify it. The app does not explicitly collect
known diabetes/treatment history, so low current labs alone cannot establish
that a person meets the source cohort's baseline diabetes-free definition.
That is an unresolved clinical applicability limitation, not a new rule.

## Repeated participants and leakage

The local saved OOF CSV has 105,740 rows, 105,740 unique participant IDs, and
zero duplicate IDs. Thus no repeated ID can cross folds in that saved file.
The source uses shuffled StratifiedKFold(5), not participant-grouped splitting,
and does not assert unique IDs. Raw-cohort duplicates, mislabeled identifiers,
and the exact training run are not independently verified. A future source
containing repeated participants would require grouped splits.

The model fits on each training fold, then predicts its held-out fold. However,
IPCW is estimated once from the entire eligible cohort before splitting. Held-out
follow-up/outcome information therefore influences training weights. This is
not a fully isolated validation pipeline; future validation should fit nuisance
steps inside folds. Bootstrap intervals resample evaluable rows with fixed
weights/predictions; they do not refit models or re-estimate censoring. The
source provides no nested model-selection audit or independent external cohort.

## Available internal metrics

Exact values below come from output/validation_report.json, copied unchanged
from the local candidate's aggregate report. The AUC, average precision and
Brier score were independently recalculated from saved OOF predictions and
weights and match. Calibration values and intervals are reported values; they
were not refit/rebootstrapped in this audit.

| Metric | Documented value |
|---|---:|
| IPCW ROC AUC | 0.9260644157084451 |
| AUC bootstrap 95% CI | [0.9198359591326516, 0.9319554293893695] |
| IPCW average precision | 0.18802159900496124 |
| IPCW Brier score | 0.010310387255022684 |
| Brier bootstrap 95% CI | [0.009856561987564402, 0.01077088115306721] |
| Calibration intercept (ideal 0) | -0.045641082114279106 |
| Calibration slope (ideal 1) | 0.9860404830250553 |
| IPCW mean predicted risk | 0.011885736776171623 |
| IPCW observed risk | 0.011825228641399309 |

The report's `kaplan_meier_risk_3y` is assigned the IPCW observed-risk value in
the source; it is not a separately computed event Kaplan-Meier estimate.
Subgroup details are in the JSON. The labels "below 100", "100-109", "110-125"
use actual mmol/L cuts 5.6 and 6.1, so the labels are approximate mg/dL bands.
Their AUCs are respectively 0.8305654446099124, 0.7105079550513526,
and 0.625878225483591. Narrow-range discrimination is materially weaker.

## Categories and reproducibility limits

Lower: raw p<0.01; Increased: 0.01<=p<0.05; Elevated: p>=0.05.
These are project-defined action bands. No threshold-specific external
calibration, decision-curve/net-benefit study, or Nigerian clinical validation
was found supporting these bands. Exact boundaries and repeat inference pass
application tests; that is software validation, not clinical validation.

train_unified_reference.py is an unchanged archival copy for code inspection.
Its original SOURCE/OUT paths refer to a separate research workspace and it
requires the absent workbook plus an Excel-reading dependency. It is not a
portable training command and must not be run to start the app. The historical
survival files retained in the original repository concern a retired model and
are omitted from this copy; they do not validate this unified model.
Runtime reproducibility is tested separately from training
reproducibility; the full original training run was not repeated.
