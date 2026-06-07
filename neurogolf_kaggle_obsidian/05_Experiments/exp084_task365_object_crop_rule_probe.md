# exp084_task365_object_crop_rule_probe

## 目的

`task365` が低cost `Slice/Gather` に近いobject crop ruleか検証する。

## 結果

- best candidate: `max_count2`
- validation: `266/266`
- train: `3/3`
- test: `1/1`
- arc-gen: `262/262`

## Rule

入力の非ゼロconnected componentをobjectとして取り、各object内の色 `2` の数を数える。色 `2` が最大のobjectを選び、そのbbox patchを出力する。

## 判断

task365はfull-pass rule hit。次はrectangle-window selectorとしてONNX loweringを試す。

## Risk

- leakage risk: low。raw lookupなし。
- overfitting risk: medium。色2最大というtask-specific ruleだが、all arc-gen full pass。
