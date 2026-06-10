# exp219_task039_bbox3x3_cost_probe

## 目的

task039 `bbox_top_left 3x3 crop` を `GatherElements` でONNX化し、baseline cost `7772` より安いか確認する。

## 結果

- validation: `264_pass_0_fail`
- candidate cost: `48219`
- baseline cost: `7772`
- status: `no_cost_gain`

## 判断

ruleは正しいが、`GatherElements` dynamic bbox cropは重すぎる。提出なし。

## リスク

- leakage risk: low
- overfitting risk: medium-low
