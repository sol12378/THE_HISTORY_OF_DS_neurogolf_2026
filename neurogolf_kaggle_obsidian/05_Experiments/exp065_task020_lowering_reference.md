# exp065_task020_lowering_reference

## 目的

`exp062` のtask020明示ruleをONNX化する前に、入力だけから `bbox5 crop`、`target color`、`class selector`、`orientation/missing`、`writeback` を再現できるか確認する。

## 結果

- target task: `task020`
- reference pass: `266/266`
- train: `3/3`
- test: `1/1`
- arc-gen: `262/262`
- status: `rule_reference_pass`

初回は `265/266` で、失敗は1件だけだった。原因はbbox中心にある単独色をtarget colorと誤認したこと。center-only singletonをtarget color候補から除外すると、raw lookupを使わずに `266/266` へ到達した。

## 解釈

task020はteacher artifactをコピーせず、非lookupの幾何/role ruleとして入力のみから再現できる。これはP0 teacher-gain taskで初めて、ONNX loweringへ進める十分なreference gateを満たした結果である。

## Leakage / Overfitting Risk

- leakage risk: low。arc-gen正解tableやteacher artifactを参照しない明示rule。
- overfitting risk: medium。task020専用ruleなので、Kaggle hidden生成で崩れないかはsingle-task delta提出で較正が必要。

## Decision

次は `exp066_task020_correctness_onnx_lowering` として、まず `266/266` passかつstrict cost `90133` 未満を狙う。cost改善したらstrict seedにtask020だけ差し替え、Kaggleへsingle-task delta提出してlocal/LB gapを測る。
