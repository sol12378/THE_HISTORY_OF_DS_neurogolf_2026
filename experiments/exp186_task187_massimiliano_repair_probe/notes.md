# exp186_task187_massimiliano_repair_probe

## 目的

exp183でpublic-zeroと推定したtask187を、exp185監査で見つけたdistinct full-local-valid rawへ差し替える。

## 結果

- status: `submitted_complete_no_gain`
- Kaggle ref: `53526998`
- public LB: `6005.90`
- base_public_lb: `6005.93`
- expected_gain_if_public_pass: `13.435307852707972`
- expected_lb_if_public_pass: `6019.365307852709`
- observed_gain_vs_base: `-0.03`
- repair_source: `experiments\exp185_risk_next4d_candidate_validation_audit\task187_01_exp002_public_blend_6500_fast.onnx`
- repair_sha256: `2ebbc64c3257fa66b2030c42845bb1b9da22d09c9369fe61e530a138b97cb46a`
- validation: `266_pass_0_fail`, reason `ok`
- repair_cost: `105313`
- zip sha256: `8a9b6f7c04f0b39d7a9e14c05dc7c5d2a7c436b73b528116a48b34096cdb4d7e`

## 判断

submitted_but_rejected。

public LB `6005.90` はbase exp178 `6005.93` を更新せず、期待 `6019.37` に届かなかった。このdistinct full-local-valid rawも公開ではrepairにならない。exp183のmissing dropはtask187由来の可能性が高いが、この候補は採用しない。

## リスク

medium-to-high: repair raw is from an existing public/source blend candidate.
medium-to-high: full local validation plus public-zero evidence may not guarantee private robustness.
