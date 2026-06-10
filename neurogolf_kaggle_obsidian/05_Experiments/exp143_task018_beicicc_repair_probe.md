# exp143_task018_beicicc_repair_probe

## 目的

exp140でpublic-zeroと推定されたtask018を、full local validationを通るbeicicc候補で単独修復できるか確認する。

## 結果

- candidate: `experiments/exp142_task018_candidate_validation_audit/task018_05_exp130_public_code_6285_floor.onnx`
- local validation: `266_pass_0_fail`
- manifest cost: `33897`
- expected LB if public pass: `5945.12`
- Kaggle ref: `53520866`
- public LB: `5930.55`
- exp127からのdelta: `0.0`

## 解釈

beicicc task018候補もpublicでは0点。full local validationだけでは task018 の public failure を修復できない。

## 判断

採用しない。残るfull-ok別rawの `exp_b035_new_source_full_arc_blend` を最後の既存source repair候補としてprobeする。

## リスク

- leakage risk: high。
- overfitting risk: high。
