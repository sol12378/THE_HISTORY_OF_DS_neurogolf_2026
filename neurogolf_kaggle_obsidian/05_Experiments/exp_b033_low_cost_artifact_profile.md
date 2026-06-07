# exp_b033_low_cost_artifact_profile

## 目的

`exp_b025_submit_safe_delta_union` のcurrent submit-safe bundleから、すでにcost 250〜600以下へ落ちているartifactをプロファイルし、次のrule/compiler loweringで真似すべき低cost patternを抽出する。

## 結果

- base: `experiments\exp_b025_submit_safe_delta_union`
- total tasks: `400`
- cost<=250: `19`
- cost<=600: `27`
- cost<=2000: `83`
- local delta: `0.0`
- submission: no submit。診断profileのみ。

## 低cost task

cost<=250 task:

`16, 53, 56, 103, 113, 116, 164, 172, 179, 210, 241, 258, 261, 276, 309, 311, 326, 337, 385`

cost<=600付近まで含めたnear-cost task:

`16, 53, 56, 67, 87, 103, 113, 116, 135, 140, 149, 164, 172, 179, 186, 210, 241, 258, 261, 276, 309, 311, 326, 337, 373, 385, 393`

## Route summary

- `crop_or_resize`: 139 tasks, cost<=250 `8`, cost<=600 `12`, median cost `11212`
- `same_shape_global_transform`: 12 tasks, cost<=250 `0`, cost<=600 `0`, median cost `64297`
- `sparse_edit_or_object_completion`: 249 tasks, cost<=250 `11`, cost<=600 `15`, median cost `19441`

## Compiler lesson

- 低cost成功例の中心は `Gather`, `Slice`, `Pad`, small `Conv`。
- `Transpose` 1 nodeや `Gather` 1 nodeのような固定変換はcost 0〜30まで落ちる。
- near-250のshape/index artifactには、`ArgMax`, `ReduceSum`, `Slice`, `Pad`, small `Conv` の小さい組み合わせが多い。
- region fill / line extensionをnaive flood-fill unrollやfull-grid indexで表す方向は、これまで通り主戦略から外す。

## 判断

次のscore-producing PDCAは、未解決のconnectivity ruleを無理にunrollするより、既存低cost artifactをtemplate dictionaryとして読み、object-anchor crop / shape-index / small local mask compilerへ寄せる。

有力な次候補:

- low-cost crop exemplars: `task103`, `task56`, `task67`, `task135`, `task149`, `task393`
- fixed `Gather`/`Slice` exemplars: `task16`, `task53`, `task116`, `task164`, `task172`, `task210`, `task276`, `task309`, `task311`, `task337`, `task385`
- small `Conv` exemplars: `task258`, `task261`

## Risk

- leakage risk: low。submitted-safe current bundleのartifact構造をprofileしただけ。
- overfitting risk: low。candidate生成や採用はしていない。
- future risk: 低cost patternを別taskへ移植する際は、full arc-gen validationとKaggle small-delta calibrationが必要。

## Outputs

- `experiments/exp_b033_low_cost_artifact_profile/result.json`
- `experiments/exp_b033_low_cost_artifact_profile/artifact_profile.csv`
