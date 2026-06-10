# exp144_task018_b035_repair_probe

## 目的

task018 の最後の既存full-ok別rawである `exp_b035_new_source_full_arc_blend` 候補で、public-zero を修復できるか確認する。

## 結果

- candidate: `experiments/exp142_task018_candidate_validation_audit/task018_06_exp_b035_new_source_full_arc_blend.onnx`
- local validation: `266_pass_0_fail`
- candidate cost: `476184`
- candidate points: `11.92644`
- expected LB if public pass: `5942.47644`
- Kaggle ref: `53520958`
- public LB: `5942.48`
- exp127からのdelta: `+11.93`

## 解釈

task018 の public-zero 修復に成功した。observed delta は expected delta と一致しており、exp140 の推論とも整合する。

## 判断

current public LB best として採用候補。ただし `exp_b035` は public/source blend 由来であり、private robustness は未証明。最終候補化にはrisk reviewが必要。

## リスク

- leakage risk: high。
- overfitting risk: high。
