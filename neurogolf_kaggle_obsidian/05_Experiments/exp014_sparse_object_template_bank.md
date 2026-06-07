# exp014_sparse_object_template_bank

## 目的

sparse/object completion向けtemplate bankを実装し、exp012 baseより低costなstatic ONNX候補を作る。

## 実装

- fixed sparse edit
- color-conditioned sparse edit
- bbox/local系の基礎
- hole/neighbor fill
- line/edge extension系の初期候補
- `signature_scatternd_lookup`
- `global_transform_color_map`
- `constant_sparse_output`

## 結果

- target task: `71`
- improved task: `62`
- 改善の中心は `signature_scatternd_lookup`

## 判断

汎用rule templateはvalidationで落ちるtaskが多い。短期的なlocal estimate改善はlookup系が強いが、private耐性は低い。次はtask286/task203/task313/task328などに、ruleを個別に当てるgraph surgeryが必要。

## Risk

arc-gen sampleを利用したlookup候補が中心なのでleakage/overfitting riskは高い。提出候補ではなく、local upper bound探索として扱う。
