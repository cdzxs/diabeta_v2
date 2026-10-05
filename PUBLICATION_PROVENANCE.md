# Publication provenance and permission findings

Updated 2026-10-05. **Owner authorship statement recorded; no conflicting
third-party claim or applicable publication prohibition identified in the reviewed
records.** The owner states: "These models and project-specific training/inference
code were developed for my DiaBeta project with ChatGPT assistance."
This is the owner's statement, not an independent authorship certification or a
new license grant. It supersedes the earlier uncertainty about an unidentified
supplier. A ZIP created or delivered during project work does not establish an
external supplier. The owner has authorized publication of the reviewed update.
No project license is selected; browser verification remains pending.

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
  training promotion are not fully documented for exact reproduction. This is a
  reproducibility gap, not evidence of conflicting third-party ownership.
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

## Current evidence assessment

The Git contributor attribution, project source, model metadata and ZIP inspection
provide no concrete conflicting third-party authorship claim for the active models
or project-specific training/inference code. Missing standalone permission files
are not treated as evidence of an external supplier or as a requirement to obtain
permission from a hypothetical party. The owner's statement supplies the previously
missing project-development context; it does not transfer ownership of NHANES or
Rich Healthcare data and does not license upstream software.

NCHS statistical-use/non-identification conditions and Dryad CC0 terms were
rechecked against the linked official sources. No term reviewed specifically bars
this research-model/code publication. Raw data and record-level outputs remain
excluded from the latest tree; their existing remote history remains intact at
the owner's instruction. The concrete Getty rights evidence concerns an excluded
image, not the active models. Upstream dependency terms remain applicable.

Remaining limitations: missing source workbook and exact training/promotion
provenance, internal evaluation/IPCW limitations, no model privacy certification,
and pending browser verification. These are documented limitations, not an
invented permission requirement. No project reuse license has been chosen.
