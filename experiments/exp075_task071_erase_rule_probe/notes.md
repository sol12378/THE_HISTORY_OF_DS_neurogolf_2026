# exp075_task071_erase_rule_probe

## 目的

`task071` が `task085` と同じ horizontal-bar erase 型なら、既知ruleを横展開して低cost置換候補にできる。

## 結果

- best: `train_changed_colors_erase_components` = `1/265`
- train: `0/2`
- test: `0/1`
- arc-gen: `1/262`
- changed_to_zero_examples: `6/265`
- changed_from_zero_examples: `0/265`

## 解釈

`task071` は `task085` 型の単純eraseではない。変更セルは mixed component edit で、0への消去だけでは説明できない。
このため、`task071` をすぐ ONNX lowering するのは避け、recolor/copy compiler または小さな component branch-tree の探索へ回す。

## Risk

- leakage risk: low。診断のみで、lookup提出物は生成していない。
- overfitting risk: low。full-pass候補なし。
