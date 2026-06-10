# exp236_task174_internal_subcrop_audit

## 目的

task174について、最大色component bbox/maskの内部sub-cropで出力を説明できるか監査する。

## 結果

- best internal subcrop: `138/266`
- full hits: none
- near hits: none
- examples without hit: `128`

## 判断

task174は単純な内部sub-cropでは解けない。より複雑な内部選択ruleが必要。

## リスク

- leakage risk: low
- overfitting risk: medium-low
