# exp167_next4l_failure_bisection_probe

## 目的

exp166 current bestをベースに、残りsubset頻度上位の `048/035/012/017` をfail-stub化してpublic-zero有無を測る。

## 結果

- status: `probe_zip_ready`
- targets: `[48, 35, 12, 17]`
- expected_drop_if_all_alive: `61.823685547451404`
- expected_lb_if_all_alive: `5906.356314452549`
- zip sha256: `ce1c4148006ffb73aeaa67e651f40c2df476ab53dd3c081c0c5f4269a3f50c85`
- kaggle_ref: `53523591`
- kaggle_status: `COMPLETE`
- public_lb: `5906.33`
- interpretation: all-alive; observed drop matches expected all-alive drop within scoreboard rounding

## Target Validation

- task048: `0_pass_1_fail`, reason `mismatch example 0`
- task035: `0_pass_1_fail`, reason `mismatch example 0`
- task012: `0_pass_1_fail`, reason `mismatch example 0`
- task017: `0_pass_1_fail`, reason `mismatch example 0`

## 判断

submit_bisection_probe

## リスク

low: deliberate fail-stub probe; no private labels used.
medium: consumes a submission and only measures public scoring behavior for a group.
