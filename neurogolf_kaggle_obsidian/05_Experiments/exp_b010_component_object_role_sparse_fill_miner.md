# exp_b010_component_object_role_sparse_fill_miner

## 目的

coordinate-firstではなく、object/component/color-role firstのsparse fill ruleを探索する。候補はrow/column reduction、directional reduction、small bbox mask、hole/component maskへloweringできるものに限定する。

## 結果

- target task count: `9`
- rule count: `13`
- evaluated candidate: `2`
- full pass hit: `2`
- full pass task: `task037`
- rule:
  - `opposite_ray_all_same_color`
  - `opposite_ray_diag_same_color`
- validation: train `3/3`, test `1/1`, arc-gen `262/262`, total `266/266`
- local estimate delta: `0.0`
- submission: `no_submit_yet`

## 解釈

初めてP0 sparse fillで説明可能ruleのfull passを得た。task037は「対角方向の同色端点に挟まれたbackground cellをその色で埋める」ruleで説明できる。

## Risk

- leakage risk: low-to-medium。train-fit ruleで、arc-gen labelは合成に使っていない。
- overfitting risk: medium。hidden生成に対してはsingle-task deltaで確認が必要。

## Decision

task037 ruleをONNX loweringへ進める。ただしloweringはfull-gridではなく、diagonal-specific small mask / reductionを狙う。
