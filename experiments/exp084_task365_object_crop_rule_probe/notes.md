# exp084_task365_object_crop_rule_probe

## 目的

`task365` が低cost `Slice/Gather` に近いobject crop ruleか検証する。

## 結果

- best: `max_count2` = `266/266`
- train: `3/3`
- test: `1/1`
- arc-gen: `262/262`

## 判断

full-passならcrop loweringへ進む。partialならobject selection featureを追加する。

## Risk

- leakage risk: low。単純rule候補のみ。
- overfitting risk: medium。task-specific selectionはall-arc full pass必須。
