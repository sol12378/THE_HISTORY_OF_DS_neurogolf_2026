# exp099_oss_optimizer_stack_probe

## 目的

革新的compiler設計の第1実験として、汎用OSS/ORT optimizerを現行submit-safe ONNXへ適用するだけで公式cost改善が出るかを測る。改善がなければ、汎用optimizerではなくNeuroGolf専用rewrite/e-graph extractorへ進む。

## 結果

- probe task count: `62`
- evaluated candidates: `0`
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.812218`
- module availability: `{'onnxsim': False, 'onnxoptimizer': False, 'onnxscript': False, 'onnx_graphsurgeon': False, 'polygraphy': False, 'egglog': False}`

## 解釈

汎用optimizerは推論速度や標準冗長削除を目的にしており、NeuroGolfの `memory+params` costを直接最小化しない。現環境では主要OSS moduleも未導入で、ORT offline optimizerは過去exp009でno gain、今回probeではtask002でプロセス終了した。次は低cost artifact miningと専用rewrite/extractionへ移す。

## Decision

`result.json` の acceptedが空なら、OSSは実行基盤・rewriter部品として使い、汎用最適化passそのものを主戦力にしない。

## Risk

- leakage risk: low。
- overfitting risk: low。
