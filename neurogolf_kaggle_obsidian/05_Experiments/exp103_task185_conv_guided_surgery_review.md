# exp103_task185_conv_guided_surgery_review

## 目的

compiler campaign #5。task185のone-Conv方針を、既存artifact surgeryと格子位置inventoryで現実評価する。5実験レビューも同時に行う。

## 結果

- task185 baseline cost: `59584`
- node count: `242`
- duplicate initializer groups: `0`
- unused nodes: `0`
- candidate:
  - `sanitize_current_task185`: `267_pass_0_fail`, cost `59584`, no gain
  - `dedupe_initializers_task185`: `267_pass_0_fail`, cost `59584`, no gain
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.812218`

## 5実験レビュー

verdict: `mixed_but_directionally_useful`

No local/LB improvement yet, but the five experiments moved from generic optimizer hopes to a NeuroGolf-specific compiler grammar and ruled out simple archetype scans. This is useful only if the next block emits candidates from computed_slice_pad/one-Conv patterns rather than continuing catalogs.

## Decision

task185は46格子位置patternがあり、static Conv proxyをそのまま提出候補にはできない。simple surgeryでも改善がないため、次blockは `computed_slice_pad` / `tiny_dynamic_shape_index` compilerとして位置推定を扱う。

## Risk

- leakage risk: low。
- overfitting risk: low〜medium。
