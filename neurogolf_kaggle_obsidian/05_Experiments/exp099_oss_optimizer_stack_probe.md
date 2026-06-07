# exp099_oss_optimizer_stack_probe

## 目的

革新的compiler設計30実験シリーズの第1実験。汎用OSS/ORT optimizerを主戦力にできるかを確認し、NeuroGolf専用compilerへ進むべきか判断する。

## 結果

- module availability:
  - `onnxsim`: false
  - `onnxoptimizer`: false
  - `onnxscript`: false
  - `onnx_graphsurgeon`: false
  - `polygraphy`: false
  - `egglog`: false
- probe task count: `62`
- evaluated candidates: `0`
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.812218`

## 解釈

現環境には主要OSS moduleが未導入。さらにORT generic optimizationは過去 `exp009` でno gain、今回のoffline serialization probeでもtask002でプロセス終了したため、30実験campaignを汎用optimizer tuningに費やすのは非効率と判断する。

## Decision

汎用ONNX optimizerを主戦力にしない。OSSは導入できる場合でも、NeuroGolf専用の低cost artifact mining / rewrite rule / e-graph extraction の部品として使う。

## Risk

- leakage risk: low。
- overfitting risk: low。
