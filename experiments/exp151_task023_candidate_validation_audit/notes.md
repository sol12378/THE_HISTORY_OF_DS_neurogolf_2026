# exp151_task023_candidate_validation_audit

## Hypothesis

task023の既存source候補の中に、public-zeroを修復できるfull validation候補がある可能性がある。

## Result

- source rows: `33`
- unique raw candidates: `5`
- full_ok_candidates: `2`

## Decision

Use a public-distinct full_ok candidate for a single-task repair probe if any candidate is not the current public-zero raw.

## Leakage / Overfitting Risk

medium-to-high: existing public/teacher candidates may be lookup-like.
medium-to-high: full local validation is not sufficient for public/private robustness.
