# exp269_farm_where_mac_zero_guardrail

## 目的

既知較正では `Where` は MAC=0 であり、cost は params + output tensor bytes で見るべきである。farm では `Where` が `CONDITIONAL_HIGH_RISK_OPS` に入っており、full-grid output という理由だけで hard reject されていたため、この1点だけを修正する。

## 変更

- `experiments/neurogolf_farm/cost_extractor.py` の `CONDITIONAL_HIGH_RISK_OPS` から `Where` を外した。
- `ScatterND` / `GatherND` / `ScatterElements` の guardrail は維持。
- `FULL_GRID_COMPOSITION` primitive の reject は今回は触らない。

## 確認

- `py_compile` 成功。
- `Where` + `ONE_NODE_DATA_MOVEMENT` + `uint8 output` は `hard_reject=false`, `cost_proxy=9000`。
- `Where` + `FULL_GRID_COMPOSITION` は `Where` 理由ではなく `full_grid_composition` 理由で reject のまま。
- `FarmRunner.run_smoke()` 成功。

## 判断

提出なし。これは farm guardrail 較正であり、exp265 public LB `6008.90` を上回る候補 bundle の local estimate 更新ではない。

## リスクと次アクション

- leakage risk: なし。
- overfitting risk: なし。
- 次は `recolor_direct(cost 44) >> recolor_cast(cost 140)` の既知較正を farm IR に表現できるよう、recolor primitive の lane を確認する。
