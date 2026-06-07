# exp029_crop_object_synthesizer

## Hypothesis

crop/resize系の一部は、trainだけから推定できる定数 `Slice` または定数 `Slice` + 1x1 `Conv` color mapで、exp023の高cost artifactより安く置換できる。

## Result

- base: `experiments\exp023_graph_surgery_exp016`
- target crop/resize tasks: `85`
- generated candidates: `7`
- improved tasks: `0`
- local estimate: `6479.398825`
- delta: `0.000000`
- gap to 6500: `20.601175`
- submission decision: `no_submit: below 6500 threshold`

## Interpretation

失敗済みのfull-grid `Tile`、large `ScatterND`、signature lookupは使わず、cost guardrail上で軽い候補だけを検査した。
改善候補が出た場合も、baseがexp023由来のlocal upper boundを含むため、submit前にはfull validationとrisk reviewが必要。

## Risks

- leakage risk: medium: train-fitted crop rules are validated on test and arc-gen sample20; base still includes high-risk lookup artifacts.
- overfitting risk: medium: crop rules may be sample-specific; no private-like family holdout yet.

## Next

6500未満なら、crop候補の次は `same_shape_global_transform` または `line_grid_fill` の小さい `Gather`/`Conv` 系に限定して探索する。
