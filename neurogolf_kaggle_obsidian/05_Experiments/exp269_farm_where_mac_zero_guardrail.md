# exp269_farm_where_mac_zero_guardrail

## 仮説

`Where` は MAC=0 のため、farm で conditional high-risk として一律 hard reject すると、recolor / mask selection 系の低cost候補を誤って捨てる可能性がある。

## 実施

- 変更対象は `experiments/neurogolf_farm/cost_extractor.py` のみ。
- `CONDITIONAL_HIGH_RISK_OPS` から `Where` を削除。
- `FULL_GRID_COMPOSITION` lane の reject は維持し、1ループ1変更を守った。

## 結果

- `Where` + `ONE_NODE_DATA_MOVEMENT` + `uint8 output`: `hard_reject=false`, `cost_proxy=9000`。
- `Where` + `FULL_GRID_COMPOSITION`: `hard_reject=true`。理由は `full_grid_composition` であり、`Where over full-grid output` ではなくなった。
- farm smoke成功。

## 判断

提出なし。tooling guardrail 較正のみで、候補bundleのlocal estimate更新ではない。

## 次アクション

次は `recolor_direct(cost 44)` と `recolor_cast(cost 140)` の差を farm 側で表現できるかを確認し、必要なら IR primitive / cost guard を1施策で追加する。
