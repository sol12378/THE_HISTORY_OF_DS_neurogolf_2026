# exp066_task020_correctness_onnx_lowering

## 目的

`exp065` のtask020入力only referenceをONNXへloweringし、strict seed task020 cost `90133` 未満のsingle-task deltaを作る。

## 結果

- status: improved
- validation: 266_pass_0_fail
- baseline cost: 90133
- candidate cost: 73080
- delta: 0.209732
- reason: ok
- Kaggle submission: ref `53417088`, LB `5930.10`
- LB delta vs exp005 strict seed: `+0.21`

## 解釈

正しさ優先で、5x5 bbox crop、色1〜9のgroup count、center-only singleton除外、corners/edge/inner group fillをONNX化した。

## Decision

strict seed single-task deltaとしてKaggleへ提出し、public LBは `5929.89 -> 5930.10` になった。local delta `+0.209732` とほぼ一致したため、この明示rule/official validation環境はKaggleでも通用する可能性が高い。
