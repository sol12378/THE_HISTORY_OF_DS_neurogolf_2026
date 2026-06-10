# exp264_current_rank321_400_fullarc_bypass_sweep

## 目的

exp263 current best bundleのcost rank321-400にも、1-pass full-arc bypass surgeryの上積みが残るかを確認する。

## 結果

- base: `experiments/exp263_exp262_partial_bypass_bundle_submit_probe/submission.zip`
- generated_candidate_count: `500`
- sample20_improved_count: `11`
- fullarc_selected_count: `8`
- selected_tasks: `299, 229, 235, 339, 129, 144, 318, 26`
- local_delta: `+0.5976596441240858`

## 判断

local estimateを更新したため、exp265で再生成・full-arc replay bundle化する。低cost側ではcostの絶対差が小さくてもlog score差が大きく、rank321-400 sweepにも提出可能なdeltaが残っていた。

## Risk

- leakage risk: low-medium。current best artifactへのgraph surgeryであり、full-arc gateは通している。
- overfitting risk: medium-low。available examples上の同値性に依存するため、replay bundleで再確認する。
