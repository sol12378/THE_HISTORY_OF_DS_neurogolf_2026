# exp050_task020_rule_probe

## 目的

task020のsignature lookup teacherを明示ruleへ圧縮するため、差分とcomponent geometryを調べる。

## 結果

- 266 examples all same-shape 10x10
- changed cellsは常に3
- 変更はすべて `0 -> target_color`
- nonzero bbox preserved
- target colorは 1/2/3/4/8 のいずれか

## 解釈

task020はsparse object completion。signature lookupを使わず、5x5付近の相対位置orbit補完として説明できる可能性が高い。
