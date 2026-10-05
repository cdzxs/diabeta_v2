# Sources, attribution, and unresolved permissions

Reviewed 2026-10-05. This inventory does not select a license for DiaBeta.
See [PUBLICATION_PROVENANCE.md](PUBLICATION_PROVENANCE.md) for the subsequent
file-by-file investigation, Git attribution, archive checks and exact outstanding
permissions. Required Stage 1 and Stage 3 redistribution remains unconfirmed.
No project LICENSE was found. The owner must confirm authorship/permission and
decide the project's licensing before offering reuse rights.

## Data-derived models

Stage 1/2: CDC/NCHS, National Health and Nutrition Examination Survey,
2013-2014, 2015-2016, and 2017-2018. Construction scripts identify the source
cycles. [NHANES citation guidance](https://wwwn.cdc.gov/nchs/NHANES/NhanesCitation.aspx)
generally describes federal data as public domain. The
[NCHS Data User Agreement](https://www.cdc.gov/nchs/policy/data-user-agreement.html)
restricts use to statistical analysis/reporting and prohibits identifying
participants or linking to identifiable records. These sources support research
reuse; they do not certify model privacy, clinical suitability, or ownership of
project code. No CDC, NCHS, HHS, or US Government endorsement is implied.

Stage 3: Chen, Ying; Zhang, Xiao-Ping; Yuan, Jie et al. (2018).
*Data from: Association of body mass index and age with incident diabetes in
Chinese adults: a population-based cohort study*. Dryad,
[doi:10.5061/dryad.ft8750v](https://datadryad.org/dataset/doi:10.5061/dryad.ft8750v).
Dryad's [terms, sections 1 and 4](https://datadryad.org/terms) provide for CC0
dataset publication and reuse, with scholarly citation expected. That supports
reuse of this source dataset and derived research, but is not a license grant
for third-party training code or DiaBeta itself. The local training workbook is
absent, so its reported hash has not been matched to a newly downloaded source.
The active artifact does match the locally supplied candidate byte-for-byte.

Raw participant datasets, row-level prediction exports, and historical backup
models are not intended release files. See PRE_COMMIT_REVIEW.md for existing
index/history exceptions. Model serialization is not proof of anonymization;
no membership-inference or model-inversion assessment was performed.

## Software and visual assets

Runtime dependencies are installed from their upstream distributions, not
vendored. Their installed license metadata/notices remain applicable; no
project license replaces those terms. The installation uses Flask, flask-cors,
joblib, NumPy, pandas, scikit-learn, SciPy, threadpoolctl and their dependencies.
The frontend requests Inter and Playfair Display from Google Fonts; font files
are not bundled. This also means the default pages are not fully offline.

The original repository retains eleven raster images with unresolved rights.
care-team.jpg has an embedded Getty Images rights-statement URL; that is not a
license grant. All eleven are omitted from the sibling publication copy, which
uses original SVG placeholders and omits personal team content/contact links.
External font requests are also omitted from that copy. No original is deleted.
Do not infer permission from file presence. Authorship and distribution permission
for required supplied model/training code remain open; see the provenance review.
