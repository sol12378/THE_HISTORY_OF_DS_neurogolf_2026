# exp162_task025_rule_diagnostic

## 目的

exp146/147でpublic-zeroかつ既存source差し替え不可と判明したtask025について、rule repairの入口として入出力差分と単純ruleを診断する。

## 結果

- example_count: `266`
- shape: 全例shape preserved
- output color: 全例で入力既存色のみ
- all_changes_on_zero_count: `0`
- simple rule probe: identity / fill / row-col fill / bbox fill / mirror / rot180 はすべて `0/266`

## 判断

task025は単純なzero-fillやglobal transformではない。既存色セルの sparse relocation/recolor 型に見えるため、次はchanged-cell target predicateと色移動構造を採掘する。

## 成果物

- `experiments/exp162_task025_rule_diagnostic/result.json`
- `experiments/exp162_task025_rule_diagnostic/notes.md`
- `experiments/exp162_task025_rule_diagnostic/task025_example_diagnostic.csv`
- `experiments/exp162_task025_rule_diagnostic/task025_rule_probe.csv`

## リスク

- leakage risk: low。提供されたtrain/test/arc-genの診断のみ。
- overfitting risk: medium。local exampleからのrule推論はpublic/private hidden variantへ過適合する可能性がある。
