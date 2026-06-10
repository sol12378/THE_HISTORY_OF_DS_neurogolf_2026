# exp316_task187_zero_component_fullarc_audit

## 目的

task187 public-zero repair の前提として、exp031 の zero-component fill rule を全 arc-gen へ拡張して再検証する。

## 結果

- status: `fullarc_rule_pass`
- rule: border-connected zero component -> color `3`
- rule: enclosed zero component -> color `2`
- train: `3_pass_0_fail`
- test: `1_pass_0_fail`
- arc-gen: `262_pass_0_fail`
- submission: なし。Python rule audit のみ。

## 判断

task187 は correctness-first ONNX repair へ進める。public-zero repair なので cost は通常改善候補ほど厳しく見ないが、full validation と static check は必須。

## リスク

- leakage risk: low-medium。色は train から推定し、arc-gen は validation のみ。
- overfitting risk: medium。Python rule full-arc pass は必要条件であり、ONNX/LB probe は未完。
