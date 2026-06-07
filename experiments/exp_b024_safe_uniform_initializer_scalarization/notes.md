# exp_b024_safe_uniform_initializer_scalarization

## 目的

seddik notebook由来のuniform initializer scalarizationをstrict seed top80 taskへ安全ゲート付きで試す。

## 結果

- safe initializer rows: 16
- generated candidates: 10
- improved tasks: [36, 51, 77, 110, 128, 284, 358, 383]
- local delta: 0.027485
- decision: submit_micro_delta_if_improved

## Risk

- leakage risk: low-to-medium: public pattern but applied to strict seed with full-arc gate.
- overfitting risk: medium: scalarization is semantic only under safe broadcast assumptions; full-arc validation required.
