# exp012_template_factory_core notes

## 目的

GPU route classifierのrankingを使い、top cost task向けにstatic ONNX templateを自動生成する。

## 結果

- input: `exp005_top_cost_rewrite_strict`
- top_k: `400`
- arc_gen_sample: `20`
- lookup_arc_gen_sample: `20`
- baseline local estimate: `6282.230228092811`
- new local estimate: `6296.297369919829`
- delta: `+14.067141827018531`
- improved task count: `38`
- adopted template: `signature_scatter_lookup`
- zip sanity: 400 files, `taskNNN.onnx` only

## 解釈

`signature_scatter_lookup` は高cost taskを中心に改善したが、20件sample-localでも6500には届かない。固定幾何template、固定mask、色写像、cropは今回の対象ではほぼ効かなかった。

## Risk

- leakage risk: 高。lookup templateはarc-gen labelを利用できるため、public/local上限探索として扱う。
- overfitting risk: 高。private耐性は期待しない。
- submit risk: 高。full validationとユーザー確認なしに提出しない。

## 次PDCA

- top cost taskのONNX graphをtask別に読み、MaxPool/Where/MatMul/Scatter系を直接置換するgraph surgeryを増やす。
- GPUで小型CNN/conv templateをtask別に学習し、strict static ONNXとしてvalid passするものだけ採用する。
- primitive別holdoutでprivate-like validationを作り、public artifact寄せの危険度を測る。
