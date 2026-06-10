# exp164_task025_line_projection_rule

## 目的

task025について、完全な縦/横ラインをguideとしてstray cellをライン隣接セルへ射影するinput-only ruleを検証する。

## Rule

- 非ゼロ色ごとに、全セルがその色で埋まったhorizontal/vertical guide lineを検出する。
- guide line自体は保持する。
- guide line外の同色セルは、同じ列または同じ行でguide lineの隣接セルへ射影する。
- guide lineを持たない色は削除する。

## 結果

- pass_count: `266/266`
- decision: `rule_found_lower_next`

## 判断

task025はrule-found。次はONNX loweringとcost validationを行う。current modelはpublic-zeroかつcost `89286` なので、full validation passかつcostが許容範囲なら単独repair submit候補になる。

## 成果物

- `experiments/exp164_task025_line_projection_rule/result.json`
- `experiments/exp164_task025_line_projection_rule/notes.md`
- `experiments/exp164_task025_line_projection_rule/task025_line_projection_eval.csv`

## リスク

- leakage risk: low。input-only geometric rule。
- overfitting risk: medium。hiddenでguide lineが完全ではないvariantがある場合に崩れる可能性がある。
