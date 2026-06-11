# exp328 task185 dynamic background detector probe

## 目的

task185 の残課題である dynamic background detector を、入力の非ゼロ最頻色として検証し、ONNX subgraph の official cost を測る。

## 仮説

背景色は入力内の非ゼロ最頻色であり、`ReduceSum` + `ArgMax` で安く検出できる。

## 結果

- detector rule: nonzero modal input color
- validation: `267_pass_0_fail`
- bg histogram: `{1: 32, 2: 23, 3: 30, 4: 25, 5: 26, 6: 26, 7: 34, 8: 37, 9: 34}`
- ONNX detector static: ok
- detector official cost: `143`
- memory / params: `132` / `11`

## 判断

dynamic bg detector は task185 の blocker ではなくなった。cost も小さいため、次はこの detector を dilated axis selector に接続し、検出された bg channel を selector/core から除外する correctness-first candidate を作る。

注意点は exp204 の巨大 `row_templates` / `col_templates` で、これをそのまま使うと cost が baseline を超える。compact start/spacing table から index を導出するか、まずは correctness-only と cost-risk を明示した candidate として切る。

## Submission

提出なし。detector subgraph probe のみ。

## Risk

- leakage risk: low。入力のみの modal-color detector。
- overfitting risk: low-to-medium。task-specific だが全 local examples で検証済み。

## Next

exp329 で detector-connected task185 dynamic axis candidate を作る。
