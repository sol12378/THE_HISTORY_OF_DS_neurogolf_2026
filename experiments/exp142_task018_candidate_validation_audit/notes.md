# exp142_task018_candidate_validation_audit

## Hypothesis

task018の既存source候補の中に、full validationを通るrepair候補が残っている可能性がある。

## Result

- source rows: `34`
- unique raw candidates: `6`
- full_ok_candidates: `2`

## Decision

Use the cheapest full_ok candidate for a single-task repair probe if any exist; otherwise task018 needs rule repair rather than existing-source replacement.

## Leakage / Overfitting Risk

medium-to-high: existing public/teacher candidates may be lookup-like.
medium-to-high: full local validation is not sufficient for public/private robustness.
