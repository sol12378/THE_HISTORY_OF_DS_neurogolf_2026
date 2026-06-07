# exp071_component_color_role_profile

## 目的

`exp070` で固定template型ではないと判定された LOCAL_PREDICATE_FILL 上位taskを、component/color-role/changed-cell roleで再分類する。

## 仮説

大量taskをcost 250〜600へ落とすには、既存artifact削減ではなく、changed-cellを説明する小さなDSL/DAGを新規合成する必要がある。上位taskをfamilyに分解できれば、次の合成器をscore直結のtaskへ集中できる。

## 結果

- targets: [173, 77, 158, 133, 71, 285, 85, 251, 383, 54]
- family_counts: {'mixed_component_edit': 3, 'multi_color_recolor_or_copy': 5, 'object_erase_or_mask_clean': 1, 'single_color_sparse_fill': 1}
- priority_rule_tasks: [85, 251]

## 解釈

`single_color_sparse_fill` と `object_erase_or_mask_clean` は、出力色/消去色が単純で、次に small predicate tree を試す価値が高い。

## リスク

- leakage risk: low。profileのみで提出物は作らない。
- overfitting risk: medium。次段のpredicate treeはbranch数を制限し、full arc-gen pass後にsingle-task deltaでLB較正する。

## 次アクション

`task251` の単色fill、または `task071/085` のerase/mask cleanを対象に、positive/negative cell feature datasetを作り、深さ2〜4のpredicate treeを合成する。
