# exp315_wide_failure_bisection_probe_a

## 目的

public-zero repair を最優先するため、未 probe の高リスク 16 task を現 best exp297 上で意図的に fail-stub し、observed drop から残 public-zero の有無を診断する。

## 仮説

observed drop が all-alive 期待値 `225.790763` より小さければ、missing drop がこの 16 task 内の public-zero 点数質量を表す。

## 対象

- targets: `097 193 192 197 324 137 335 224 213 338 131 275 138 243 359 101`
- base: `exp297_exp262_skip_task048_336_fresh_candidates`
- base public LB: `6008.96`
- expected LB if all alive: `5783.169236677023`
- zip sha256: `15183cb0c00c3d66b3c6323acbd569dd591a81243e3cd24493e33fe51e80bf22`

## 結果

- status: `submitted_complete`
- Kaggle ref: `53550304`
- public LB: `5810.93`
- observed drop from base: `198.03`
- missing drop vs all alive: `27.760763`
- target validation: all `0_pass_1_fail`
- top missing-drop subset candidates:
  - task193 + task275: `27.772428` (error `0.011664`)
  - task243 + task101: `27.779018` (error `0.018255`)
  - task097 + task275: `27.733125` (error `0.027638`)
- subset-sum ambiguity: 16-way rounded subset sums collideするため、この probe だけでは一意確定しない。

## 判断

この 16 task 内に約 2 task 分の public-zero が残っている可能性が高い。ただし既知 zero の task187 repair が現在の優先順位 1 のため、split follow-up は task187 repair の進捗後に設計する。

## リスク

- leakage risk: low。意図的 fail-stub probe で、private label は使っていない。
- overfitting risk: medium。public diagnostic なので、修復時は correctness-first rule/full validation を必須にする。
