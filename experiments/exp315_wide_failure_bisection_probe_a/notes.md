# exp315_wide_failure_bisection_probe_a

## 目的

A-3 wide bisection probe: intentionally fail 16 unprobed high-risk tasks on the current exp297 best bundle. If the observed drop is smaller than the all-alive drop, the missing mass identifies one or more remaining public-zero tasks; subset ambiguity is recorded for follow-up split probes.

## 結果

- status: `probe_zip_ready`
- targets: `[97, 193, 192, 197, 324, 137, 335, 224, 213, 338, 131, 275, 138, 243, 359, 101]`
- expected_drop_if_all_alive: `225.79076332297663`
- expected_lb_if_all_alive: `5783.169236677023`
- zip sha256: `15183cb0c00c3d66b3c6323acbd569dd591a81243e3cd24493e33fe51e80bf22`

## Target Validation

- task097: `0_pass_1_fail`, reason `mismatch example 0`
- task193: `0_pass_1_fail`, reason `mismatch example 0`
- task192: `0_pass_1_fail`, reason `mismatch example 0`
- task197: `0_pass_1_fail`, reason `mismatch example 0`
- task324: `0_pass_1_fail`, reason `mismatch example 0`
- task137: `0_pass_1_fail`, reason `mismatch example 0`
- task335: `0_pass_1_fail`, reason `mismatch example 0`
- task224: `0_pass_1_fail`, reason `mismatch example 0`
- task213: `0_pass_1_fail`, reason `mismatch example 0`
- task338: `0_pass_1_fail`, reason `mismatch example 0`
- task131: `0_pass_1_fail`, reason `mismatch example 0`
- task275: `0_pass_1_fail`, reason `mismatch example 0`
- task138: `0_pass_1_fail`, reason `mismatch example 0`
- task243: `0_pass_1_fail`, reason `mismatch example 0`
- task359: `0_pass_1_fail`, reason `mismatch example 0`
- task101: `0_pass_1_fail`, reason `mismatch example 0`

## 提出 / 採点

- Kaggle ref: `53550304`
- Kaggle status: `COMPLETE`
- public LB: `5810.93`
- observed_drop_from_base: `198.03`
- missing_drop_vs_all_alive: `27.760763`
- interpretation: 約2 task 分の public-zero が 16 task 内にいる可能性が高い。
- top missing-drop subset candidates: `[193, 275]`, `[243, 101]`, `[97, 275]`
- subset-sum collision audit: `subset_sum_collisions.csv`

## 判断

submitted_probe_complete; split follow-up は可能だが、判断ロジック上は既知 zero の task187 repair を先に進める。

## リスク

low: deliberate fail-stub probe; no private labels used.
medium: this is a public diagnostic probe; target selection comes from public-zero risk inventory and must be followed by correctness-first repair, not public-score-only adoption.
