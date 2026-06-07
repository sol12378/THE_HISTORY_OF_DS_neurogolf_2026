# exp060_task020_orientation_audit

## Hypothesis

task020の失敗はD4 orientation復元にある。

## Result

- examples: `266`
- ambiguity counts: `unique_orientation: 266`
- compatible variant count: always `1`
- true class oracle output: `266/266`

## Interpretation

orientationはtemplate classが分かれば一意に決まる。したがって残課題はorientationではなく、3-template class selectorである。

## Decision

次は入力target color位置からcanonical template classを選ぶ規則を探す。
