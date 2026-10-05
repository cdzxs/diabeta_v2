# DiaBeta 2.0 local publication review

## Current release checkout review (2026-10-05)

This section supersedes the older folder/status descriptions below. A fresh clone
of https://github.com/cdzxs/diabeta_v2 main at
514be70a6cf6c1792ae6fbd0868ff3a83ce906f5 received the 52 verified publication
files and additional ignore rules. This checkout retains existing remote history.
The inventoried 27 NHANES XPT files, constructed participant CSV, record-level
prediction CSV, historical SHAP explanations, model backup, retired artifacts and
uncleared raster assets are removed from the latest tree. Originals and a separate
local preservation copy remain intact. Historical copies remain reachable on
GitHub; this normal cleanup does not erase them. No force-push/history rewrite.

Application changes include the unified Stage 3 inference integration, input and
sex-code validation, server-enforced Stage 2 eligibility, missing-glucose blocking,
stale-result invalidation, escaped patient text and documented research limitations.
The model artifacts and thresholds are unchanged from the verified candidate.

Fresh checks on this checkout: 18 backend tests (including both Flask startup
methods and /health), 45 hybrid checks, 5 Stage 2 frontend scenarios, 43 Stage 3
checks and 6 release scenarios passed. The previously isolated runtime installation
is reused; no new clean-environment installation is claimed for this checkout.
Browser/visual verification remains pending; automated JavaScript tests do not
replace the manual checklist. No training or new clinical validation occurred.

The 52 staged paths match PUBLICATION_MANIFEST.txt. Every application, test and
model file is byte-identical to the verified publication copy; only this review
and .gitignore differ. The final tree contains no CSV/XPT/XLSX/ZIP, key files,
database files, research directory or environment. Text scanning found no common
GitHub/OpenAI/AWS token or private-key patterns. This pattern scan is not an
exhaustive credential or personal-data detector; the two serialized active model
files are hash-matched, not certified free of embedded training information.
The existing history retains the previously documented data and scan limitations.
Dependency consistency passes pip check. Staged diff whitespace checks pass with
CRLF endings explicitly recognized; existing source line endings were preserved.

**Verdict: complete local release commit; not cleared for push.** Required artifact
rights remain unresolved. Stage 1 model and training/diagnostic sources are
Git-attributed to Ezeji Godswill, but no verified authorship/rights-holder grant
covers redistribution. The unified Stage 3 model, inference and supplied training
reference match the delivered ZIP, but no identified author or redistribution
license/permission accompanies that delivery. These are missing-evidence blockers,
not a demonstrated prohibition in the NHANES or Dryad dataset terms. Dataset
public-use/CC0 status does not establish rights in these contributed implementations.
No new permission is inferred from this cleanup request. See PUBLICATION_PROVENANCE.md
for exact paths and source chain. A project license has not been selected.

The intended publication tree is PUBLICATION_MANIFEST.txt. It excludes the separate
research archive, raw data, participant outputs, environments, credentials and
local preservation folders. The following older audit is retained as historical
evidence; its original Git status is not the release checkout's current status.

Updated 2026-10-05. **Publication verdict: Not ready.**
The clean copy is prepared and software checks pass, but redistribution rights
for required models/code remain unresolved. Browser verification is pending.
The owner explicitly confirmed that permission cannot yet be established.

## Original repository versus publication copy

| Location | Role and current status |
|---|---|
| `../diabeta_v2 (original repository)` | Original Git repository; 28 reachable commits preserved; existing staged/unstaged work and all local data retained |
| `. (this publication folder)` | Separate sibling folder; 52 intended files; no .git directory, inherited history, remote configuration or virtual environment |

Only two original index entries were changed in this continuation, using
`git rm --cached`. The previously staged work was not restaged or overwritten.
New reports and documentation edits are working-tree changes. No commit, push,
publication, history rewrite, retraining, threshold change or deletion of local
data/model artifacts occurred. No external application API was exercised.

The original repository is NOT the clean publication source. Its old history
still contains datasets and retired artifacts. The sibling folder avoids that
history without modifying it. Required models are retained in this local candidate
for testing only; its existence does not authorize their public redistribution.

## Resolved items

### Exact historical data identified

See [HISTORY_DATA_INVENTORY.md](HISTORY_DATA_INVENTORY.md) for all exact paths,
including serialized artifacts and source backups. Across all refs: 28 commits,
118 distinct paths, 153 unique blobs. The 29 tabular-data paths comprise:

- 27 NHANES XPT files in `DPM/data/`: BMX, BPQ, DEMO, DIQ, GHB, GLU, MCQ,
  PAQ and SMQ, each with H/I/J suffixes. These are public-use participant-level
  health/demographic survey data, not identified local DiaBeta patient exports
  or credentials. All 27 parsed schemas include SEQN. NCHS statistical-use and
  non-identification restrictions still apply; no restricted-access NHANES file
  was identified. Public-use health data are not treated as harmless to publish.
- `DPM/data/stage1_v2_nhanes_8predictors.csv`: 16,327 participant-level derived
  NHANES records, with public survey IDs/cycle, predictors and labels.
- `DPM/models/stage1_v2_test_predictions.csv`: 3,266 generated per-record
  evaluation rows: labels and probabilities, without explicit patient names/IDs.

Historical fitted models, scalers and feature/race mappings are generated binary
artifacts. SHAP values are per-record derived explanations. No claim is made that
these binaries cannot contain sensitive training information. No restricted dataset
or real patient assessment export was identified by this inventory/schema review.
No participant rows or credentials were printed or reproduced in the reports.

### Two unwanted index artifacts removed; local copies preserved

| Path | Classification | Result |
|---|---|---|
| DPM/models/stage1_v2_baseline_before_integration.pkl | Historical fitted-model backup; generated binary | Removed from index; ignored; original bytes match HEAD |
| DPM/models/stage1_v2_test_predictions.csv | Generated row-level evaluation output | Removed from index; ignored; original content matches HEAD after Windows line-ending normalization |

Both exclusions were already present in the working .gitignore, so no additional
ignore edit was needed. All 51 LOCAL_ONLY_FILES.txt paths remain on disk.
All 21 original .pkl/.joblib artifacts outside verification environments/copies
retain their prior SHA-256 hashes. The index hash is unchanged after these two
authorized removals; no other index entry was changed during this continuation.

### Separate publication copy prepared

The explicit inventory below contains 52 files, including both unchanged active
model artifacts, inference, backend/frontend code, runtime dependencies, tests,
research evidence and permission notices. It contains no CSV/XPT/XLSX data,
patient export/database, environment, cache, backup, delivery ZIP or .git history.

Eleven original raster assets are excluded. The exact paths and evidence are in
[PUBLICATION_PROVENANCE.md](PUBLICATION_PROVENANCE.md). `care-team.jpg` contains
an XMP rights-statement URL pointing to Getty Images; no corresponding permission
grant/invoice was found. Other assets have no substantiated redistribution evidence.
Git image-upload commits and metadata markers do not establish rights.

New `frontend/publication-mark.svg` and `frontend/publication-illustration.svg`
use simple original geometric shapes. They do not trace or embed excluded images.
HTML/CSS references were updated; two initially broken CSS replacement references
were corrected and the final link scan passes. Personal portraits, biographies,
contact email links and external font requests are omitted from the copy.
The About Project page remains navigable. The original frontend is preserved.

Optional `backend/model/race_map.pkl` is omitted; tested built-in labels are used.
Retired Stage 3 training source/metadata/report are omitted, including a
workstation-specific dataset path. The originals and first copied versions are
preserved in the original workspace/audit archive. Active model logic is unchanged.

## Permissions investigation and remaining blockers

1. **Required Stage 1 rights unresolved.** The active model exactly matches the
   artifact introduced by commit `9a3f952`, attributed to Ezeji Godswill. Its
   regularized-classifier parameters match the diagnostic/validation records.
   Git attribution and parameter correspondence are not proof of original
   authorship or permission. No license/permission grant was found. The owner
   must identify the code/model rights holders and provide a redistribution
   license or written permission covering source and serialized weights,
   including institutional/employer constraints and the candidate promotion record.
2. **Required Stage 3 rights unresolved.** The model and training reference match
   `DiaBeta_unified_Rich_3year_model_v1.zip`; the local candidate is the immediate
   source. The ZIP has no author/license notice or archive comment. Loaded model
   metadata supplies no authorship/license field. Identify who created/delivered
   the ZIP, its source/contract, and rights-holder permission for weights,
   inference and training code. Source-data authorship is not model authorship.
3. **Project licensing undecided.** No project license was found in history or
   selected. Confirm contributions/permissions before deciding the project's
   license. Possession or AI assistance does not establish permission.
4. **Browser verification pending.** Retried the browser runtime; discovery
   returned an empty list. No real walkthrough, responsive layout or print/PDF
   verification was possible. [MANUAL_BROWSER_CHECKLIST.md](MANUAL_BROWSER_CHECKLIST.md)
   covers both entry paths, High/missing-glucose blocks, edits/in-flight changes,
   reset/cross-tab behavior, records and literal patient-text rendering.

Source terms were checked against CDC/NCHS guidance and the Rich dataset's Dryad
record/terms. They support research reuse of the public source data under the
stated terms, not blanket clearance of the derived implementation. Precise links,
authorship evidence, affected files and information still needed from the owner
are in PUBLICATION_PROVENANCE.md and THIRD_PARTY_NOTICES.md. Optional original
image permissions no longer block this stripped copy, unless those assets are restored.

## Clean-copy verification — actual results

Fresh Windows Python 3.14.0 venv with system site packages disabled was created
outside the publication folder, in the original repository's ignored
`.precommit-check/publication-audit/venv`. Installation used the publication
copy's `backend/requirements-runtime.txt` and succeeded. Direct versions:
Flask 3.1.3, flask-cors 6.0.5, joblib 1.5.3, NumPy 2.4.6, pandas 3.0.3,
scikit-learn 1.8.0, SciPy 1.18.0, threadpoolctl 3.6.0. Resolved dependency
list/test logs remain local in the audit directory; no environment is shipped.

| Check, run against the sibling publication folder | Result |
|---|---|
| Fresh installation and pip check | Passed; no broken requirements |
| Backend unittest discovery | 18 passed |
| Hybrid assessment checks | 45 passed |
| Stage 2 frontend handler tests | 5 scenarios passed |
| Stage 3 frontend/cache tests | 43 passed |
| Release state/handoff tests | 6 scenarios passed; 24 sex/restoration combinations |
| Root `flask --app backend.app run` | Started; /health HTTP 200 with status=ok |
| Backend-folder `flask --app app run` | Started; /health HTTP 200 with status=ok |
| Documented frontend http.server | /index.html HTTP 200 with expected content |
| Inline JavaScript syntax | 13 script blocks parsed |
| Local HTML/CSS asset links | No missing targets |
| Required runtime paths and models | Present; model hashes match originals |
| Original models/index preservation | All 21 model hashes match; only two authorized index removals |
| Browser/visual/print | Pending; no connected browser |

Tests use real serialized inference plus synthetic inputs; mocked classifier
outputs exercise every High-producing rule. Automated coverage includes Stage 1
through Stage 3, direct Stage 2, required fields/units/sex conversion, every High
block and missing fasting glucose, deterministic predictions, stale/current-state
separation, edited/in-flight requests, new assessments, saved snapshots and escaped
patient text. Lightweight DOM/storage substitutes do not certify actual browser
rendering. No real browser records were accessed. Temporary servers were stopped.

Original clinical rules/thresholds and all model bytes remain unchanged. The new
geometric placeholders have not been visually approved. Other Python versions,
operating systems, fully offline installation and model retraining were not tested.

## Secrets and private-data checks

- Original reachable history: all 99 decodable UTF-8 text blobs scanned for
  common token/private-key/literal-credential patterns; zero matches. 54 binary
  or non-UTF-8 blobs were skipped by that scan. All 118 reachable paths inventoried.
- Publication copy: all 50 text files scanned for the same patterns plus email
  and workstation-user paths; no matches after the exclusions. The two remaining
  binary files are the required active models. No raw/row-level dataset or saved
  patient record is included. Synthetic test identities are intentional.
- The original repository's data/history is not declared scrubbed. Unreachable
  objects, reflogs, external repositories and arbitrary/encoded secrets are outside
  scope. Binary/model privacy was not exhaustively assessed; no membership-inference
  or inversion test was performed. A heuristic scan is not proof of absence.

## Stage 3 evidence limits retained

This is a research prototype, not a diagnosis or a clinically validated Nigerian
risk calculator. The model is the unified approximately three-year Rich cohort
classifier, not the retired survival model. Its SHA-256 is
`110600c8ec947a3c5003a1680332af1ae2dec4b7ce6ac4e0c66ee416284d9d2e`.
The Stage 1 hash is
`5d523c6f46a7f6a76be2a05f2a1bea4ddffd19f2df880e2e91caf9c63bbf5923`.

The saved OOF file previously inspected has 105,740 unique IDs and zero repeated
IDs; no raw workbook is present to independently verify full-cohort/source identity.
Training uses row-based StratifiedKFold with IPCW estimated before splitting from
the full eligible cohort. Therefore fully isolated/leakage-free validation is not
established. Baseline known diabetes/treatment status is not explicitly collected
by the application. Nigerian external validation and clinical utility evidence
for the project-defined <1%, 1%-<5%, >=5% categories remain unavailable.

The aggregate report and EVIDENCE_REVIEW.md retain exact documented metrics:
IPCW AUC 0.9260644157084451; average precision 0.18802159900496124;
Brier 0.010310387255022684; calibration intercept -0.045641082114279106;
slope 0.9860404830250553. The prior audit recomputed AUC/AP/Brier from saved OOF
predictions; no refitting/retraining or new clinical validation was done here.
Calibration/CI values remain documented evidence, not newly verified clinical claims.

## Publication inventory

The sibling PUBLICATION_MANIFEST.txt lists these exact files. Required artifacts
are included conditionally for local testing; unresolved permission still blocks upload.

```text
.gitignore
DPM/assessment/hybrid_assessment.py
DPM/data/construct_stage1_nhanes.py
DPM/data/construct_stage1_v2_nhanes.py
DPM/data/stage1_v2_construction_report.txt
DPM/models/diagnose_stage1_v2.py
DPM/models/stage1_v2_diagnostic_report.txt
DPM/models/stage1_v2_model.pkl
DPM/models/stage1_v2_training_report.txt
DPM/models/stage1_v2_validation_report.txt
DPM/models/train_stage1_v2.py
DPM/models/validate_stage1_v2.py
DPM/stage3/unified_rich_model/EVIDENCE_REVIEW.md
DPM/stage3/unified_rich_model/MODEL_CARD.md
DPM/stage3/unified_rich_model/inference.py
DPM/stage3/unified_rich_model/output/unified_rich_3y_model.joblib
DPM/stage3/unified_rich_model/output/validation_report.json
DPM/stage3/unified_rich_model/train_unified_reference.py
DPM/tests/test_hybrid_assessment.py
HISTORY_DATA_INVENTORY.md
LOCAL_ONLY_FILES.txt
MANUAL_BROWSER_CHECKLIST.md
PRE_COMMIT_REVIEW.md
PUBLICATION_MANIFEST.txt
PUBLICATION_PROVENANCE.md
README.md
THIRD_PARTY_NOTICES.md
backend/STAGE2_RULE_AUDIT.md
backend/app.py
backend/audit_stage2_rules.py
backend/requirements-runtime.txt
backend/requirements.txt
backend/stage2_explanation.py
backend/test_release_validation.py
backend/test_stage3_integration.py
backend/test_startup.py
frontend/about.html
frontend/about_me.html
frontend/app.js
frontend/gauge.js
frontend/index.html
frontend/publication-illustration.svg
frontend/publication-mark.svg
frontend/records.html
frontend/result.html
frontend/stage1.html
frontend/stage2.html
frontend/stage3.html
frontend/style.css
frontend/tests/release.test.cjs
frontend/tests/stage2.test.cjs
frontend/tests/stage3.test.cjs
```

## Final original Git status

72 staged changed paths, 16 paths with unstaged edits, 9 untracked files (counts overlap). The original index has 55 files. Only the two named removals changed it in this continuation.

```text
MM .gitignore
D  DPM/assessment/hybrid_assessment_before_v2_model_swap.py
D  DPM/assessment/predict_model2.py
D  DPM/data/BMX_H.xpt
D  DPM/data/BMX_I.xpt
D  DPM/data/BMX_J.xpt
D  DPM/data/BPQ_H.xpt
D  DPM/data/BPQ_I.xpt
D  DPM/data/BPQ_J.xpt
D  DPM/data/DEMO_H.xpt
D  DPM/data/DEMO_I.xpt
D  DPM/data/DEMO_J.xpt
D  DPM/data/DIQ_H.xpt
D  DPM/data/DIQ_I.xpt
D  DPM/data/DIQ_J.xpt
D  DPM/data/GHB_H.xpt
D  DPM/data/GHB_I.xpt
D  DPM/data/GHB_J.xpt
D  DPM/data/GLU_H.xpt
D  DPM/data/GLU_I.xpt
D  DPM/data/GLU_J.xpt
D  DPM/data/MCQ_H.xpt
D  DPM/data/MCQ_I.xpt
D  DPM/data/MCQ_J.xpt
D  DPM/data/PAQ_H.xpt
D  DPM/data/PAQ_I.xpt
D  DPM/data/PAQ_J.xpt
D  DPM/data/SMQ_H.xpt
D  DPM/data/SMQ_I.xpt
D  DPM/data/SMQ_J.xpt
D  DPM/data/stage1_v2_nhanes_8predictors.csv
D  DPM/models/diabeta_dataset1_model.pkl
D  DPM/models/diabeta_dataset2_model.pkl
D  DPM/models/stage1_v2_baseline_before_integration.pkl
D  DPM/models/stage1_v2_test_predictions.csv
D  DPM/stage3/predict_risk.py
D  DPM/stage3/stage3_discrete_survival_model.json
D  DPM/stage3/stage3_discrete_survival_model.pkl
D  DPM/stage3/stage3_preprocessor.pkl
D  DPM/stage3/stage3_threshold.txt
D  DPM/stage3/stage3_xgboost_model.pkl
AM DPM/stage3/unified_rich_model/MODEL_CARD.md
A  DPM/stage3/unified_rich_model/inference.py
A  DPM/stage3/unified_rich_model/output/unified_rich_3y_model.joblib
D  DPM/stage3_baseline_backup/predict_risk.py
D  DPM/stage3_baseline_backup/stage3_preprocessor.pkl
D  DPM/stage3_baseline_backup/stage3_threshold.txt
D  DPM/stage3_baseline_backup/stage3_xgboost_model.pkl
AM LOCAL_ONLY_FILES.txt
AM PRE_COMMIT_REVIEW.md
MM README.md
A  backend/STAGE2_RULE_AUDIT.md
MM backend/app.py
D  backend/app_before_stage2_v2_model_swap.py
A  backend/audit_stage2_rules.py
D  backend/model/features_nhanes_binary.pkl
D  backend/model/model_binary_nhanes.pkl
D  backend/model/scaler_nhanes_binary.pkl
D  backend/model/shap_values_binary.pkl
D  backend/model/threshold_bounds_nhanes.pkl
A  backend/requirements-runtime.txt
A  backend/stage2_explanation.py
A  backend/test_stage3_integration.py
AM backend/test_startup.py
MM frontend/about.html
 M frontend/app.js
MM frontend/index.html
MM frontend/result.html
 M frontend/stage1.html
MM frontend/stage2.html
D  frontend/stage2_before_v2_model_swap.html
MM frontend/stage3.html
AM frontend/tests/stage2.test.cjs
AM frontend/tests/stage3.test.cjs
?? DPM/stage3/unified_rich_model/EVIDENCE_REVIEW.md
?? DPM/stage3/unified_rich_model/output/validation_report.json
?? DPM/stage3/unified_rich_model/train_unified_reference.py
?? HISTORY_DATA_INVENTORY.md
?? MANUAL_BROWSER_CHECKLIST.md
?? PUBLICATION_PROVENANCE.md
?? THIRD_PARTY_NOTICES.md
?? backend/test_release_validation.py
?? frontend/tests/release.test.cjs
```
