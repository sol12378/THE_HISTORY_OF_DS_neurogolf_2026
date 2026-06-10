# exp146_next4c_failure_bisection_probe

## Hypothesis

既知aliveとtask018を除外した次点consensus group `021/014/025/008` のpublic scoring状態をfail-stub probeで測る。

## Result

- status: `probe_complete_partial_zero`
- targets: `[21, 14, 25, 8]`
- expected_drop_if_all_alive: `57.12814711961373`
- expected_lb_if_all_alive_from_exp127: `5873.421852880387`
- zip sha256: `7e17c8a8c9972e49a31f131413353230855775a2f47e875b5798ae128f7ef551`

## Target Validation

- task021: `0_pass_1_fail`, reason `mismatch example 0`
- task014: `0_pass_1_fail`, reason `mismatch example 0`
- task025: `0_pass_1_fail`, reason `mismatch example 0`
- task008: `0_pass_1_fail`, reason `mismatch example 0`

## Decision

Kaggle ref `53521220` として提出し、public LB は `5887.02`。exp127からのdrop `43.53` は、task021 + task014 + task008 の点数和と一致し、task025 の `13.6004` 点が欠けている。

task025 はpublic-zeroの可能性が非常に高い。次はtask025 repair候補を監査する。

## Leakage / Overfitting Risk

low: deliberate fail-stub probe; no private labels used.
medium: consumes a submission and only measures public scoring behavior for a group.
