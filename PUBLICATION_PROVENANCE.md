# Publication provenance and permission findings

Reviewed 2026-10-05. **Required-model redistribution remains unresolved.**
The owner explicitly cannot confirm permission yet. Possession, a Git commit,
model parameters, dataset access, or AI assistance is not a permission grant.
The sibling publication folder is a local candidate, not approved for upload.

## Active Stage 1 model and code

Affected files: `DPM/models/stage1_v2_model.pkl`,
`DPM/models/train_stage1_v2.py`, `DPM/models/diagnose_stage1_v2.py`,
`DPM/models/validate_stage1_v2.py`, and associated construction/report files.

- Git records the model and training source in commit `9a3f952`, dated
  2026-09-12, attributed to **Ezeji Godswill**. This identifies the repository
  contributor, not independently verified authorship, employment rights, or
  permission to redistribute incorporated work.
- Current model bytes match the committed artifact. SHA-256:
  `5d523c6f46a7f6a76be2a05f2a1bea4ddffd19f2df880e2e91caf9c63bbf5923`.
- It is a scikit-learn Pipeline (ColumnTransformer + HistGradientBoostingClassifier).
  Parameters include max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
  random_state=42. These match the regularized candidate described by
  diagnose_stage1_v2.py and stage1_v2_validation_report.txt, rather than proving
  that the baseline training script alone produced the active bytes.
- The diagnostic report describes a separately saved candidate and the validation
  report names that candidate. The exact candidate-to-active promotion and the
  human author/permission chain are not recorded sufficiently to establish rights.
- No author/license/provenance field was found in the loaded pipeline's top-level
  metadata. No project license or permission file exists in the 28 reachable commits.
- Dataset: public-use NHANES 2013-14, 2015-16 and 2017-18, as identified by the
  construction scripts and report. CDC/NCHS is the dataset source, not evidence
  that CDC authored or endorsed this classifier/code. See source terms below.

## Unified Stage 3 model and training code

Affected required files:
`DPM/stage3/unified_rich_model/output/unified_rich_3y_model.joblib` and
`DPM/stage3/unified_rich_model/inference.py`.
Related source: `DPM/stage3/unified_rich_model/train_unified_reference.py`,
MODEL_CARD.md and output/validation_report.json.

- Local source chain: `DiaBeta_unified_Rich_3year_model_v1.zip`,
  `stage3_model_research/unified_candidate/unified_rich_model/`, then active
  `DPM/stage3/unified_rich_model/`. The model matches the delivery ZIP and
  candidate byte-for-byte; the training reference matches the ZIP's train_unified.py.
- Model SHA-256:
  `110600c8ec947a3c5003a1680332af1ae2dec4b7ce6ac4e0c66ee416284d9d2e`.
  It is a four-predictor HistGradientBoostingClassifier. No top-level author,
  copyright, license or provenance field was found.
- No author/license notice or archive comment was found in the ZIP. The README,
  model card and training script do not identify a rights holder or grant. These
  additions are not in reachable committed history; Git cannot identify their
  original author. The archive's filename and timestamps are not authorship proof.
- Dataset is the Rich Healthcare cohort deposited by Chen et al. on Dryad,
  doi:10.5061/dryad.ft8750v. Chen et al. are source-data authors; no evidence
  establishes that they authored this particular supplied model or training code.
  The original local workbook is absent, so its reported hash is unverified.
- Evaluation limitations in EVIDENCE_REVIEW.md remain unchanged: internal OOF
  evaluation, full-cohort IPCW before folds, no proven clinical validity in Nigeria.

## Source-data terms, distinct from model/code ownership

- [NHANES citation guidance](https://wwwn.cdc.gov/nchs/NHANES/NhanesCitation.aspx)
  generally permits reproduction of federal data. The
  [NCHS agreement](https://www.cdc.gov/nchs/policy/data-user-agreement.html)
  limits use to statistical analysis/reporting and prohibits re-identification
  and linkage to identifiable data. These are public-use files, not an identified
  restricted-access dataset. This supports data research reuse, not a copyright
  grant over contributed training code or proof of fitted-model ownership.
- [Dryad dataset record](https://datadryad.org/dataset/doi:10.5061/dryad.ft8750v)
  identifies the Rich cohort and deposit. [Dryad terms](https://datadryad.org/terms),
  sections 1 and 4, provide CC0 dataset publication/reuse and expect citation.
  This does not establish who owns the supplied derived implementation.
- No raw participant datasets or row-level prediction exports are included in
  the candidate. Their removal does not itself clear fitted-model rights/privacy.

## Optional assets excluded from the publication copy

All eleven raster assets are retained in the original repository but omitted
from the sibling copy. Git records additions/updates with generic image/UI commit
messages, not rights grants. PNG text metadata supplied no license evidence.

| Exact original path | Evidence / resolution in publication copy |
|---|---|
| frontend/ai-healthcare.jpg | No permission record found; omitted |
| frontend/background.jpg | No permission record found; reference replaced with original SVG |
| frontend/care-network.jpg | No permission record found; reference replaced with original SVG |
| frontend/care-team.jpg | XMP rights statement links to Getty Images EULA; no invoice/license grant found; omitted |
| frontend/clinical-ai-hero.png | No permission record found; replaced with original SVG |
| frontend/diabeta-logo.png | No author/license record found; replaced with original geometric mark |
| frontend/health-background.jpg | No permission record found; reference replaced with original SVG |
| frontend/kenneth.png | Portrait, rights/consent unconfirmed; omitted |
| frontend/muna.jpeg | Portrait, rights/consent unconfirmed; omitted |
| frontend/physician-analytics.jpg | EXIF marker present; no licensing evidence found; replaced with original SVG |
| frontend/profile-photo.png | Portrait, rights/consent unconfirmed; omitted |

The [Getty Images agreement](https://www.gettyimages.com/eula) distinguishes a
license from mere download and restricts standalone redistribution. An embedded
rights URL is not proof of a purchased license or permission to place the raw
image in a public repository. No claim of infringement is made; the copy simply
excludes the asset. No existing images were edited or deleted.

`frontend/about_me.html` is replaced in the copy with a neutral project page;
personal biographies, portraits and contact links are omitted. The contact email
link in `frontend/about.html` is also omitted. External Google Fonts requests
are removed from the copy, using existing system-font fallbacks. Replacements
`frontend/publication-mark.svg` and `frontend/publication-illustration.svg` are
new, simple geometric code written for this task, not traced from original assets.
This does not select an overall project license or clear unrelated code rights.

The optional `backend/model/race_map.pkl` is also omitted; application-defined
labels are used. Retired `DPM/stage3/train_stage3_survival.py`,
`DPM/stage3/stage3_survival_metadata.json` and
`DPM/stage3/stage3_survival_training_report.txt` are omitted from the copy.
They concern a different model, and the training script contains a local
workstation-specific dataset path. Original files and the initial copied versions
are preserved locally; no retired training was run.

## Exact information still needed from the owner

1. For Stage 1: identify the original authors of the training/diagnostic code and
   fitted model, confirm whether the Git-attributed contributor owns those rights,
   and provide the applicable license or written permission allowing public
   distribution of both source and serialized weights, including any employer or
   institutional restrictions. Supply the candidate promotion/training provenance
   record if available.
2. For Stage 3: identify who delivered/created the ZIP, its original source URL
   or agreement, and a license or explicit rights-holder permission covering the
   fitted weights, inference and training code. Identify any additional source
   code/assets incorporated by that supplier. A statement that the cohort is open
   is not a substitute.
3. Once contributions are cleared, decide the project's own license. No license
   has been chosen on the owner's behalf. Optional original photos/logo need no
   clearance for this stripped copy; permissions are needed only to restore them.

Until items 1 and 2 are resolved, the required artifacts remain in the local
candidate solely for testing and the publication verdict remains **Not ready**.
