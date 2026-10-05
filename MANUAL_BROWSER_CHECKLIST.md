# Manual browser verification — pending

Browser discovery returned no connected browser on 2026-10-05. Automated API
and page-handler checks do not replace this checklist. Use the sibling
publication copy and synthetic inputs only. Follow its README to start both
local servers. Use a separate browser profile; do not clear real patient records.

1. Open `http://127.0.0.1:8000/index.html`. Check desktop/mobile layouts,
   navigation and the new geometric placeholders; confirm no broken images.
2. Choose Start New Assessment, then Stage 1. Enter name `Synthetic A`, age 45,
   Female, BMI 27, race code 3 (Non-Hispanic White), no family history, no
   hypertension, active, never smoker. Submit and check the result/person summary.
   Continue to Stage 2; Female must remain selected. Enter HbA1c 5.9% and fasting
   glucose 100 mg/dL, submit, then generate Stage 3. Check the three-year result,
   research disclaimer and category. Repeat unchanged inputs: same probability.
3. Start a new assessment directly from Stage 2 and enter the same synthetic
   profile/labs. It should work without a Stage 1 result. Stage 1 result navigation
   must not show an old score attached to this patient's details.
4. Set HbA1c to 6.5% or glucose to 126 mg/dL. Stage 2 must be High and Stage 3
   unavailable, including direct navigation to stage3.html with a previously
   cached result. Automated tests separately cover all three High-producing rules.
5. Use HbA1c 5.9% with fasting glucose blank. Stage 2 may complete; Stage 3 must
   require glucose and must not restore an old result. Zero/invalid labs must fail.
6. Edit name, sex, BMI or labs after results exist; old results must disappear.
   Edit/reset while a request is running and in a second tab; no late response
   may overwrite the new assessment. Returning between pages must not mix inputs.
7. Start New Assessment: current identity, labs and results clear, but historical
   Stage 1 records remain. Submit `Synthetic B`; Records should show one record
   per Stage 1 completion. View/export each record and verify its own identity
   and score; editing the current assessment must not alter prior records.
8. For patient name/address use literal `<b>Synthetic text</b>` and then literal
   `<img src=x onerror=alert(1)>`. Summaries/records must display text, not execute
   markup or show an alert. Use no real personal information.
9. Print a fresh and restored Stage 3 result: inspect page layout, name, labs,
   horizon and disclaimer. A blocked/stale result must not print. Inspect browser
   console for errors and confirm requests target the local backend during tests.

Record browser/version, viewport, date, pass/fail and screenshots using synthetic
data only. All items remain pending until someone actually completes them.
