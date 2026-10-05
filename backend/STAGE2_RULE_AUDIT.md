# Stage 2 rule audit

Local read-only audit, 2026-09-29. Rules, model artifacts and eligibility were not changed.
Reproduce from the project root with `python -B backend/audit_stage2_rules.py`.

## Data and provenance

The local eight-predictor NHANES dataset has 16,327 rows. Reconstructed the
documented stratified 20% test split with seed 42 (3,266 records). Joined HbA1c
and fasting glucose from GHB/GLU H, I and J XPT files by cycle and SEQN, with
one-to-one join checks. The split's labels and baseline probabilities match
the saved test CSV. That CSV does NOT match the active model: it predates the
regularized candidate integration. All counts below use newly computed active
model probabilities, not the stale CSV probabilities. Diagnostic training code
uses the same split for the regularized candidate. No model was fitted.

Active model SHA256:
`5d523c6f46a7f6a76be2a05f2a1bea4ddffd19f2df880e2e91caf9c63bbf5923`

This is an unweighted retrospective policy replay, not observed app usage,
population prevalence, prospective clinical validation or an independent Nigerian
cohort. Model selection previously inspected this test set, so it is not a new
untouched external evaluation. Missing predictors are imputed by the model;
complete-profile counts are separately identified below.

## Results

Normal for this audit means a positive HbA1c below 5.7% and/or a positive fasting
glucose below 100 mg/dL. Missing tests are never counted as measured normal tests.
The partial-lab denominator requires at least one result and every available
result below its threshold. Diagnostic lab overrides are applied before Rule 2;
Rule 3 is applied before Rule 4. There are 117 records without either lab, excluded
from the Stage 2 decision counts because the API requires at least one test.

| Measure | Rule 2: P(High) >= 70% | Rule 4: fallback P(High) >= P(Low) |
|---|---:|---:|
| High with both labs normal (515 records) | 0 / 515 (0%) | 37 / 515 (7.18%) |
| High with all available labs normal (1,487 records; includes partial labs) | 1 / 1,487 (0.07%) | 150 / 1,487 (10.09%) |
| All final High decisions attributed to this rule | 4 | 150 |
| Above High decisions with Stage 3-compatible age, BMI and labs | 2 | 32 |
| Same, with all eight profile fields present | 2 | 31 |

Thus 154 replayed records receive a model-driven High and are blocked by policy.
Of these, 34 (2 + 32) otherwise satisfy the implemented Stage 3 age/BMI/lab
boundaries; 33 have all eight profile fields present. The otherwise-compatible
population is 1,205 records. These are attributable current-policy exclusions,
not predictions of how many would be admitted by removing just one rule: after
removing Rule 2, Rule 4 could still act on some records. Other people among the
154 already lack required glucose or fall outside the Stage 3 evidence range.

Normal labs do not prove absence of diabetes: 2 of the 37 Rule 4 normal-both-lab
records report known diabetes (DIQ010=1). One of the 32 otherwise-compatible
Rule 4 records reports known diabetes. Treatment could explain normal labs, but
medication effects were not evaluated. Do not label these counts false positives
or automatically admit them to incident-diabetes prediction.

## Evidence for the policy

The hybrid source and README specify 70% and a High/Low fallback. Rule 4 ignores
the Moderate probability and is not a 50% absolute-risk cutoff: High can win with
30% High, 20% Low and 50% Moderate. The diagnostic/behavioral reports evaluate
classification and synthetic-profile behavior, not the net benefit of these
eligibility gates. The regularized candidate's reported test macro-F1 is 0.5454,
macro ROC-AUC 0.7493, High precision 0.5315 and recall 0.4690. These whole-model
metrics do not validate either gate or establish diagnosis in a person.

No local threshold-specific calibration study, decision-curve analysis, justified
clinical cost tradeoff, independent external validation, or prospective evidence
was found supporting 70% or the High-versus-Low comparison as a reason to block
Stage 3. The old README's binary-model rationale is not validation for this
active three-class model. The project bands must not be called clinical
diagnostic thresholds.

For clinical context, [NIDDK's tests and diagnosis guidance](https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis)
describes laboratory testing and usually a second test for confirmation; it does
not endorse these project ML gates.

## Missing evidence and proposed revision (not implemented)

Actual app-user exclusion counts require versioned assessment logs, full submitted
inputs, final rule, and unique encounter identifiers; these are not available in
the evaluation files. Generalization and benefit require independent Nigerian
clinical data, confirmed baseline diabetes status and treatment history, repeated
or adjudicated labs, longitudinal outcomes, calibration by subgroup and a
prospectively specified referral/eligibility utility analysis.

Recommend separating a profile-based review flag from lab findings and from
incident-risk eligibility. Model-only concern should prompt discussion/testing,
not be presented as a diagnosis. Before changing the gate, evaluate allowing
otherwise eligible people with model-only flags to receive a clearly labelled
research estimate, while preserving exclusions for known diabetes, diabetes-range
labs pending clinical review, missing required predictors and evidence limits.
Choose any revised threshold using prespecified benefit/harm criteria and external
validation, rather than choosing a new arbitrary probability. Clinician review of
discordant results and known-diabetes status is essential. The current rule order
and every-High block remain unchanged in this task.
