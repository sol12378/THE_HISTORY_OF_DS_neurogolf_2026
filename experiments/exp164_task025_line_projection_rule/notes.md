# exp164_task025_line_projection_rule

## 目的

task025について、完全な縦/横ラインをguideとしてstray cellをライン隣接セルへ射影するinput-only ruleを検証する。

## 結果

- pass_count: `266/266`
- decision: rule_found_lower_next

## リスク

low: input-only geometric rule over provided train/test/arc-gen examples.
medium: full-line assumption may fail if hidden variants use broken or partial guide lines.
