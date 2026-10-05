# DiaBeta 2.0

DiaBeta is a research-stage diabetes assessment aid. It supports discussion with a healthcare professional; it does not diagnose diabetes or replace clinical assessment.

## Run locally on Windows (PowerShell)

Start in this publication directory. Python 3.14 with the versions below is the locally tested environment. Node.js is needed only for frontend/integration tests, not to serve the app.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements-runtime.txt
.\.venv\Scripts\python.exe -m flask --app backend.app run --host 127.0.0.1 --port 5000
```

Alternatively, start the backend from its own folder (do not run both servers at once):

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m flask --app app run --host 127.0.0.1 --port 5000
```

In a second terminal, from this publication directory:

```powershell
.\.venv\Scripts\python.exe -m http.server 8000 --bind 127.0.0.1 --directory frontend
```

Open http://127.0.0.1:8000/index.html. Backend health: http://127.0.0.1:5000/health.
Use the same hostname and port consistently: browser storage is scoped to an origin.
The Flask development server is for local use, not production deployment.

The runtime requirements pin Flask, flask-cors, joblib, NumPy, pandas, scikit-learn, SciPy and threadpoolctl. The older backend/requirements.txt contains research/training dependencies, including Gunicorn; it is not the Windows runtime installation list. The runtime dependency list is verified in an isolated Windows Python 3.14 environment; see PRE_COMMIT_REVIEW.md. Only load trusted pickle/joblib files.

## Three-stage flow

1. **Stage 1: Screening.** Eight non-laboratory predictors: age, sex, BMI, race/ethnicity, family history, hypertension, physical activity and smoking. The displayed percentage is the profile model's probability for its High class, not a future diabetes probability.
2. **Stage 2: Assessment.** Enter the profile and at least one measured lab result (HbA1c or fasting glucose). Direct entry is supported. The result separates lab findings, the profile-based screening score and the final recommendation. Changing labs alone does not change the profile score.
3. **Stage 3: Future Risk.** Eligible assessments use the unified approximately three-year model with age, sex, BMI and fasting glucose. HbA1c is not a Stage 3 predictor. An entered fasting glucose is required. Every final Stage 2 High result blocks Stage 3, and the backend independently reassesses the inputs.

Stage 2 rule order remains: diabetes-range lab override; profile High probability at least 70%; prediabetes-range lab rule; High-versus-Low model fallback. See backend/STAGE2_RULE_AUDIT.md for the current policy's evidence limitations. No eligibility policy was changed in this cleanup.

Stage 3 categories use the raw probability: below 1% Lower estimated risk; 1% to below 5% Increased risk; at least 5% Elevated risk. These are project-specific action bands, not diagnostic thresholds. The page shows a rounded percentage, category, next step and applicable evidence warnings. The PDF button opens the browser print dialog; select Save as PDF.

The About page contains the simple patient guide. The detailed calculation accordion has been removed.

## Runtime files that must accompany the app

- DPM/models/stage1_v2_model.pkl
- DPM/assessment/hybrid_assessment.py
- DPM/stage3/unified_rich_model/inference.py
- DPM/stage3/unified_rich_model/output/unified_rich_3y_model.joblib
- backend/stage2_explanation.py
- Built-in race labels (the optional backend/model/race_map.pkl is omitted from this copy)

Do not ignore all .pkl/.joblib files or the model directories. The unified model directory must accompany any eventual publication once its rights are cleared. Do not run training scripts to start the app.

## Saved assessments and privacy

Assessment state is stored in browser localStorage. Editing Stage 2 invalidates its previous result; Stage 3 results are tied to the current assessment snapshot. A different browser/origin does not share those records. Calculation requests go to the configured backend. Localhost uses the local backend; non-local frontend hosts currently use the hosted URL in the frontend source. Review this before any deployment. Do not assume browser storage guarantees privacy.

Opt-in diagnostics contain health/profile data even though names and addresses are omitted. Keep DIABETA_TRACE unset and the browser diabeta_debug key absent unless debugging. Do not commit exported assessments, screenshots containing patient information, browser storage dumps or diagnostic logs.

## Verification

From this publication directory, with Node.js available on PATH:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s backend -p test_stage3_integration.py
.\.venv\Scripts\python.exe -B -m unittest discover -s backend -p test_startup.py
.\.venv\Scripts\python.exe -B DPM/tests/test_hybrid_assessment.py
node frontend/tests/stage2.test.cjs
node frontend/tests/stage3.test.cjs
```

Additional release regressions:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s backend -p test_release_validation.py
node frontend/tests/release.test.cjs
```

Optional read-only research audit: run backend/audit_stage2_rules.py with Python. It needs the local NHANES CSV/XPT files, historical prediction CSV and baseline artifact; these are not intended release files and are not needed to start the app. Saved test predictions refer to the older model, so the audit recomputes predictions with the active artifact. Training scripts are research references, not startup commands.

## Research limitations

Stage 1/2 use a non-laboratory model developed with NHANES data plus project rules. The profile probability is not a diagnosis; the model-driven Stage 2 gates lack threshold-specific clinical validation. Stage 3 uses a Chinese longitudinal health-check cohort with internal validation only. External clinical validation for Nigerian patients has not been established. The horizon is approximately three years because outcomes were detected at follow-up visits. A Research estimate badge is not a claim of clinical validation; caution warns about sparse training-cohort coverage. Predictions may repeat for nearby values in the tree-based model.

## Publication candidate status

This standalone `diabeta_v2_publication` copy contains no inherited .git history.
It is local and **not approved for publication**: required model/code permissions
remain unresolved and browser verification is pending. See PRE_COMMIT_REVIEW.md,
PUBLICATION_PROVENANCE.md and MANUAL_BROWSER_CHECKLIST.md.

Original photographs/logo and personal team content are excluded; simple original
SVGs replace image references. External fonts are removed. The optional race-map
pickle is omitted; the backend uses its built-in labels. Models/rules are unchanged.
Historical research scripts may need data available only in the original workspace.
No Git operation is needed to run this copy. Do not publish the original history.

## Provenance and licensing

See [Stage 3 evidence review](DPM/stage3/unified_rich_model/EVIDENCE_REVIEW.md)
and its aggregate validation report for exact OOF metrics and limitations.
Saved OOF participant IDs are unique, but the training splitter does not enforce
participant grouping and IPCW weights use the full cohort before splitting.
The original source workbook is absent; full training reproducibility and
leakage-free validation have not been established. The app also does not collect
known diabetes/treatment status, so model applicability must not be inferred
from below-threshold labs alone. Categories remain project-defined.

[Third-party notices](THIRD_PARTY_NOTICES.md) records data attribution, source
terms and unresolved asset permissions. No project license has been selected.
Public availability of source data is not proof that all bundled code/images
may be redistributed. The original repository's Git history retains excluded
datasets/backups; this publication folder contains no Git history.
