# exp163_task025_motion_predicate_mining

## 目的

task025の差分を色別の削除/追加ペアとして見て、cell motionやrecolorの規則性を採掘する。

## 結果

- example_count: `266`
- color_counts_preserved: `64`
- nonzero_count_preserved: `64`
- dominant_line_type_counts: horizontal `139`, vertical `117`
- balanced_all_changed_colors: `64`

## 判断

task025は任意recolorではなく、縦/横guide lineへのaxis-aligned projectionが有力。多くの例では余剰stray cellを削除し、一部だけをline隣接セルへ移動している。

## 成果物

- `experiments/exp163_task025_motion_predicate_mining/result.json`
- `experiments/exp163_task025_motion_predicate_mining/notes.md`
- `experiments/exp163_task025_motion_predicate_mining/task025_motion_features.csv`

## リスク

- leakage risk: low。提供されたtrain/test/arc-genの診断のみ。
- overfitting risk: medium。motion signatureをinput-only ruleへ落とせない場合はlocal generated examplesへの過適合になる。
