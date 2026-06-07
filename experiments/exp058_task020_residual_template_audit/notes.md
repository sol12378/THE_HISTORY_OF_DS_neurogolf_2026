# exp058_task020_residual_template_audit

## 目的

task020でD4 orbit ruleが漏らす106例について、真の変更セルtemplateをbbox-local座標で分類する。

## 結果

- examples: 266
- unique canonical templates: 3
- d4 fail template count: 3

## Top Templates

| rank | count | d4 pass | d4 fail | template |
|---:|---:|---:|---:|---|
| 1 | 92 | 56 | 36 | `((0, 2), (2, 0), (2, 4))` |
| 2 | 87 | 49 | 38 | `((0, 0), (0, 4), (4, 0))` |
| 3 | 87 | 55 | 32 | `((1, 1), (1, 3), (3, 1))` |

## Decision

この結果はrule設計用の診断であり、template tableとしてONNXへ出してはいけない。次は少数の幾何templateをobject-role条件で選択できるかを調べる。
