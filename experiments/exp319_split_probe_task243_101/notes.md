# exp319_split_probe_task243_101

## 目的

Split follow-up for exp315 after exp318 proved task193/task275 alive: fail-stub task243 and task101 only. If the LB drop is near 27.7790 they are alive; if the drop is near 0 or partial, this pair contains public-zero task(s).

## 結果

- status: `probe_zip_ready`
- targets: `[243, 101]`
- expected_drop_if_all_alive: `27.779018459001815`
- expected_lb_if_all_alive: `5981.180981540998`
- zip sha256: `25ba6679eb5d4d68808eb854e59d6de9c8eff01e9780d5673b3b6c4de2091756`

## Target Validation

- task243: `0_pass_1_fail`, reason `mismatch example 0`
- task101: `0_pass_1_fail`, reason `mismatch example 0`

## 判断

submit_probe_after_sanity; do not submit another bisection probe until this score completes

## リスク

low: deliberate fail-stub probe; no private labels used.
medium: public diagnostic split of exp315 ambiguity; any repair must use full validation and correctness-first candidates, not public-score-only adoption.
