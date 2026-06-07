# exp066_task020_correctness_onnx_lowering

## 目的

`exp065` のtask020入力only referenceをONNXへloweringし、strict seed上でtask020のcostを下げるsingle-task deltaを作る。

## 結果

- target task: `task020`
- validation: `266_pass_0_fail`
- strict task020 cost: `90133`
- candidate cost: `73080`
- local delta: `+0.209732`
- strict seed local: `6282.230228 -> 6282.439960`
- submission: ref `53417088`
- LB: `5930.10`
- LB delta vs exp005 strict seed: `+0.21`

## 実装

- dynamic bboxから5x5 local frameを作る。
- 色1〜9についてcorners/edge/inner/center/total countを計算する。
- center-only singletonをtarget color候補から除外する。
- `corners_eq1 -> corners`, `edge_mid_eq1 -> edge_mid`, `has_inner -> inner` の順にgroupを選ぶ。
- 選んだ4点groupをtarget color one-hotで上書きする。

## 解釈

これは初めてのofficial-validな非lookup rule ONNX improvementである。gainは小さいが、teacher artifactではなく明示ruleでKaggle較正へ進めた点が重要。local delta `+0.209732` に対してLB deltaも `+0.21` だったため、この検証環境は少なくともこのsingle-task deltaではKaggleとよく一致した。

## Leakage / Overfitting Risk

- leakage risk: low。teacher artifactやarc-gen answer tableを使わない。
- overfitting risk: medium。task020専用ruleなので、LBでのdelta確認が必要。

## Decision

次は同じbbox5 group-fill compilerをtask020類似のL1/L2 sparse fillへ横展開するか、task020自身をさらにcost `73080 -> <=600` へ削る。
