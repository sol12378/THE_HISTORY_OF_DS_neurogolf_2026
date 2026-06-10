# exp219_task039_bbox3x3_cost_probe

## 目的

exp218でfull hitしたtask039 `bbox_top_left 3x3 crop` をONNX化し、baseline cost `7772` より安いか確認する。

## 結果

- validation: `264_pass_0_fail`
- status: `no_cost_gain`
- candidate_cost: `48219`
- baseline_cost: `7772`

## 判断

improvedならbundle化して提出候補。no_cost_gainまたはvalidation failureなら、このdynamic bbox crop loweringは現状保留。

## リスク

- leakage risk: low。入力nonzero bboxだけを使う。
- overfitting risk: medium-low。arc-gen全例pass前提だが、hidden shape edgeは提出較正で確認する。
