# exp060_task020_orientation_audit

## 目的

task020の3-template selectorで残るD4 orientation問題を分離して診断する。

## 結果

- examples: 266
- ambiguity counts: {'unique_orientation': 266}
- compatible variant count histogram: {1: 266}
- unique true full variants: 14

## Selector Results

| selector | variant acc | output pass |
|---|---:|---:|
| first_lexicographic | 0/266 | 266/266 |
| min_missing_lexicographic | 0/266 | 266/266 |
| prefer_variant_containing_input | 0/266 | 266/266 |
| oracle_if_unique | 0/266 | 266/266 |

## Decision

orientationが曖昧なら、入力色位置からorientation featureを追加する。true full variant tableをそのまま提出してはいけない。
