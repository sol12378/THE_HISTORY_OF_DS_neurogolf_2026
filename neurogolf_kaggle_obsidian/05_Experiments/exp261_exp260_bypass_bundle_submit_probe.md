# exp261_exp260_bypass_bundle_submit_probe

## 目的

exp260でfull-arc passした27件のrank141-220 bypass candidatesを、current best exp259 bundleに積んでKaggle提出で較正する。

## 結果

- base: `experiments/exp259_exp258_bypass_bundle_submit_probe/submission.zip`
- source_manifest: `experiments/exp260_current_rank141_220_fullarc_bypass_sweep/selected_manifest.csv`
- selected_count: `27`
- failures: `[]`
- local_delta: `+0.8400867054431931`
- expected_public_lb_if_calibrated: `6007.780086705443`
- zip_sanity: `400 files`, names OK
- Kaggle ref: `53532418`
- public LB: `6007.78`

## 判断

提出完了。期待値と一致してcurrent public bestへ採用する。次はrank221+ sweepまたは低cost lowering候補の準備へ進む。

## リスク

- leakage risk: low-medium。full-arc gated graph surgery bundle。
- overfitting risk: medium-low。public LB転写は確認済みだが、private robustnessは最終統合時に確認する。
