# exp_b013_five_experiment_review_b007_b011

## 目的

user requirementに従い、exp_b007-b011の5実験が `cost<=250` / `7700` / LB最大化に本当に有意義だったかを問い直す。

## レビュー対象

- `exp_b007_bbox_local_role_miner`
- `exp_b008_shape_conditioned_sparse_fill`
- `exp_b009_bbox_affine_formula_miner`
- `exp_b010_component_object_role_sparse_fill_miner`
- `exp_b011_task037_diag_ray_lowering`

## 判定

`meaningful_but_not_score_producing`

## 有意義だった点

- b007-b009でcoordinate-first grammarを棄却できた。
- b010でtask037の説明可能rule `opposite_ray_diag_same_color` が `266/266` full passした。
- b011で正しいruleでもfull-grid Conv visibility loweringは `441,976+` costになり、strict baseline `63,726` に負けると実測できた。

## 不十分だった点

- score deltaは `0.0`。
- task037 ruleを `cost<=250` / `cost<=600` に落とすloweringはまだない。
- b010 full passは1 taskのみで、P0 sparse fill全体には広がっていない。

## 次PDCA

1. task037はrule探索ではなくlow-cost lowering問題として扱う。
2. full-grid Convはguardrail入り。diagonal-specific small mask、triangular Gather、または既存artifact surgeryで<=600、理想<=250を狙う。
3. b010 rule bankは他taskのpartial evidenceを残すように拡張し、単一rule full passだけでなくdecision tree合成へ進める。

## Submission

b011はcost gainなし、b012 teacher deltaはfull arc-gen validation failed (`24_pass_1_fail`) なのでsubmitなし。
