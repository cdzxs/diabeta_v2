# DiaBeta unified three-year diabetes-risk model

Read [EVIDENCE_REVIEW.md](EVIDENCE_REVIEW.md) for the 2026-10-05 audit,
exact aggregate metrics, artifact hash, source attribution, and validation
limitations. Reported OOF performance is not evidence of clinical validation.
The saved OOF IDs are unique, but splitting is row-based and censoring weights
were estimated before splitting. Full leakage prevention is not established.

## Intended use

Research-stage estimation of incident diabetes risk by approximately three
years for adults with fasting plasma glucose below the diagnostic threshold.
It is a risk aid, not a diagnosis or substitute for clinical assessment.

## Data and outcome

- Open Rich Healthcare longitudinal cohort
- 211,783 eligible baseline records after excluding implausible FPG and
  diabetes-range FPG
- 105,740 participants with observable three-year status
- 1,763 diabetes events by three years
- Diabetes onset was detected at follow-up visits and is interval-censored

## Predictors

- Age
- Sex
- BMI
- Fasting plasma glucose

HbA1c is not a model predictor because it was unavailable in this cohort. It
is used only to block future-risk scoring at the diabetes-range threshold.

## Method

Monotonic histogram gradient boosting. Age, BMI and fasting glucose are
constrained so increasing them cannot reduce estimated risk. Right censoring
is handled with inverse-probability-of-censoring weights at the fixed
three-year horizon.

## Internal validation

Five-fold out-of-fold validation:

- IPCW AUC: 0.926 (bootstrap 95% CI 0.920–0.932)
- IPCW Brier score: 0.0103 (95% CI 0.00986–0.01077)
- Calibration intercept: -0.046 (ideal 0)
- Calibration slope: 0.986 (ideal 1)
- Mean predicted three-year risk: 1.189%
- IPCW observed three-year risk: 1.183% (the report's Kaplan–Meier label is not a separate estimate)

FPG subgroup AUCs were 0.831 below 100 mg/dL, 0.711 at 100–109
mg/dL, and 0.626 at 110–125 mg/dL. Narrow-range subgroup
discrimination is weaker even though group-level calibration remains close.

## Evidence boundaries

- FPG: approximately 50.5 to below 126 mg/dL
- Age: 20–99 years
- BMI: 15.0–52.7 kg/m²
- Outer 1% values receive a caution warning
- FPG >=126 mg/dL or HbA1c >=6.5% blocks future-risk scoring
- Fasting glucose is required

## Limitations

- Internal validation only
- Chinese health-check cohort; external transportability to Nigerian and
  other populations is unproven
- HbA1c, physical activity and diet were unavailable
- Outcome timing is visit-detected rather than exact
- The model supports only an approximately three-year horizon
- Performance must not be compared with another score unless evaluated on the
  same independent cohort
