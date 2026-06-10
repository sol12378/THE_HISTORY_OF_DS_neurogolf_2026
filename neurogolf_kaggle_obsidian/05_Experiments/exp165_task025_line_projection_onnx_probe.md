# exp165_task025_line_projection_onnx_probe

## 目的

exp164で見つかったtask025 guide-line projection ruleをcorrectness-first ONNXに落とし、full validationとofficial costを測る。

## 結果

- validation_status: `266_pass_0_fail`
- baseline_cost: `89286`
- candidate_cost: `491600`
- status: `no_cost_gain`
- decision: `do_not_adopt_refine_lowering`

## 解釈

ruleはONNXで表現できたが、9色それぞれにfull-grid reduction / MatMul / bool maskを展開したためmemory costが大きすぎる。task025 repairの方向は正しいが、このcandidateは採用・提出しない。

## 次

- 色loopを畳んでguide colorだけを処理する表現を探す。
- full-grid bool中間を減らす。
- 既存artifactの中にguide-line検出や投影に近いsubgraphがないかsurgery候補を見る。

## 成果物

- `experiments/exp165_task025_line_projection_onnx_probe/result.json`
- `experiments/exp165_task025_line_projection_onnx_probe/notes.md`
- `experiments/exp165_task025_line_projection_onnx_probe/candidate_eval.csv`
- `experiments/exp165_task025_line_projection_onnx_probe/candidate.onnx`

## リスク

- leakage risk: low。input-only rule。
- overfitting risk: medium。hiddenでguide lineが完全ではないvariantがある場合は崩れる可能性がある。
