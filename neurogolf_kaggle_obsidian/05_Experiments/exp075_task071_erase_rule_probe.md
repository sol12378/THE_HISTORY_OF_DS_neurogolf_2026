# exp075_task071_erase_rule_probe

## 目的

`task071` が `task085` と同じ horizontal-bar alternate erase 型なら、既知ruleを横展開して低cost置換候補にできるか確認する。

## 結果

- best candidate: `train_changed_colors_erase_components`
- validation: `1/265`
- `task085_horizontal_3row_bar_middle_alternate_erase`: `0/265`
- changed_to_zero_examples: `6/265`
- touched component count: all examples touched `1` component

## 判断

`task071` は `task085` 型の単純eraseではない。変更はmulti-color recolor/copyを含むため、bar erase loweringへ進めない。

## Risk

- leakage risk: low。診断のみで提出候補なし。
- overfitting risk: low。full-pass candidateなし。

## Next

`task071` はrecolor/copy compilerまたはcomponent branch-treeへ回す。
