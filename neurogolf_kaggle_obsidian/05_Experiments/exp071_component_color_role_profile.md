# exp071_component_color_role_profile

## 目的

`exp070` で固定template型ではないと判定された LOCAL_PREDICATE_FILL 上位taskを、component/color-role/changed-cell roleで再分類する。

## 結果

- targets: `173, 77, 158, 133, 71, 285, 85, 251, 383, 54`
- family_counts:
  - `mixed_component_edit`: 3
  - `multi_color_recolor_or_copy`: 5
  - `object_erase_or_mask_clean`: 1
  - `single_color_sparse_fill`: 1
- priority_rule_tasks: `85, 251`

## 解釈

`task251` は全例 `0 -> 1` の単色sparse fillで、次にpositive/negative zero-cell datasetから小さなpredicate treeを合成する価値が高い。

`task085` は全例 output `0` へのerase/mask cleanで、消去対象componentの色・周期・行列位置をdecision tree化する候補。

`task077, 158, 383, 54` などmulti-color系は、fill compilerではなくrecolor/copy compilerへ分岐する。

## リスク

- leakage risk: low。profileのみで提出ONNXやcase tableは生成していない。
- overfitting risk: medium。次段のpredicate treeでarc-genに合わせすぎる危険があるため、branch数とfeature数を制限し、full arc-gen pass後にsingle-task deltaでLB較正する。

## 次アクション

`task251` または `task085` を対象に、changed-cell positive/negative feature datasetを作る。まず深さ2〜4のpredicate treeをtrainで合成し、full arc-gen passした候補だけONNX loweringへ進める。
