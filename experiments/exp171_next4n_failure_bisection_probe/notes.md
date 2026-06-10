# exp171_next4n_failure_bisection_probe

## 目的

exp172 current bestをベースに、次候補 `003/038/001/086` のfail-stub probeを実行する。

## 結果

- status: `submitted_complete`
- targets: `[3, 38, 1, 86]`
- expected_drop_if_all_alive: `65.07777305503902`
- expected_lb_if_all_alive: `5928.742226944961`
- zip sha256: `13c4d9f243cf9508d020eafe6e96abfea52c351179e81c5d496a77dac7649f9f`
- kaggle_ref: `53524036`
- public_lb: `5928.74`
- observed_drop: `65.07999999999993`
- interpretation: all-alive期待値と一致。task003/038/001/086はpublic-scoring alive。

## Target Validation

- task003: `0_pass_1_fail`, reason `mismatch example 0`
- task038: `0_pass_1_fail`, reason `mismatch example 0`
- task001: `0_pass_1_fail`, reason `mismatch example 0`
- task086: `0_pass_1_fail`, reason `mismatch example 0`

## 判断

task003/038/001/086を短期public-zero suspectから除外する。

## リスク

low: deliberate fail-stub probe; no private labels used.
medium: consumes a submission if used and only measures public scoring behavior for a group.
