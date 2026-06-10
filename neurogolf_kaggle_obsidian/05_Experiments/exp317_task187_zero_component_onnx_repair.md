# exp317_task187_zero_component_onnx_repair

## 目的

task187 の public-zero を、exp316 で full-arc pass した zero-component rule の correctness-first ONNX lowering で修復する。

## 仮説

task187 は border-connected zero component を color `3`、enclosed zero component を color `2` にする rule で full-arc pass しているため、unrolled boundary flood-fill ONNX が public でも通れば 0 点から約 `+12.216` を回収できる。

## 結果

- best candidate: `boundary_flood_fill_bg0_30`
- validation: `266_pass_0_fail`
- cost: `356431`
- points_if_public_alive: `12.216104048283515`
- expected_public_lb_if_pass: `6021.176104048283`
- zip sanity: `400` files / `names_ok=true`
- Kaggle ref: `53550409`
- Kaggle status: `PENDING`

## 判断

public-zero repair なので cost は高いが、0 点からの回収が期待できるため提出した。採点完了後、LB が `6021.18` 付近なら current best を更新する。

## リスク

- leakage risk: low-medium。task187 は public-zero probe で選定されたが、candidate は train 推定 + full-arc validation の input-derived rule。
- overfitting risk: medium。task-specific unrolled flood-fill であり、public LB probe による確認が必要。
