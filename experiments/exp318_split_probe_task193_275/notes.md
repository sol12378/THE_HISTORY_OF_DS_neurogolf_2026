# exp318_split_probe_task193_275

## 目的

Split follow-up for exp315: fail-stub task193 and task275 only. If the LB drop is near 27.7724 these tasks are public-scoring alive; if the drop is near 0 or partial, this pair contains public-zero task(s).

## 結果

- status: `probe_zip_ready`
- targets: `[193, 275]`
- expected_drop_if_all_alive: `27.77242767718689`
- expected_lb_if_all_alive: `5981.187572322813`
- zip sha256: `58f3a1e257385d05e82078232f4eccebf1568cbc0471dd7da72eeb67d0385b55`

## Target Validation

- task193: `0_pass_1_fail`, reason `mismatch example 0`
- task275: `0_pass_1_fail`, reason `mismatch example 0`

## 判断

submit_probe_after_sanity; do not submit another bisection probe until this score completes

## リスク

low: deliberate fail-stub probe; no private labels used.
medium: public diagnostic split of exp315 ambiguity; any repair must use full validation and correctness-first candidates, not public-score-only adoption.
