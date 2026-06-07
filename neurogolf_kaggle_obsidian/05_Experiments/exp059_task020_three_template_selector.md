# exp059_task020_three_template_selector

## Hypothesis

task020は入力geometryから3つのcanonical templateを選べば解ける。

## Result

- selector count: `5`
- full pass hits: `0`
- best selector: `touch_rule`
- best pass: `142/266`
- train/test: `2/3`, `1/1`
- arc-gen: `139/262`

## Interpretation

3-template化は有効だが、canonical templateを選ぶだけでは足りない。失敗には `template_ok_but_apply_fail` が多く、D4 orientationの復元が別問題として残っている。

## Decision

ONNX loweringなし。次はtemplate class selectorとorientation selectorを分離して探索する。full passしたらsingle-task deltaとしてKaggle較正提出する。
