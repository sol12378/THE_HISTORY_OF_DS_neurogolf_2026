# exp029_crop_object_synthesizer

## Hypothesis

crop/resize系の一部は、trainだけから推定できる定数 `Slice` または定数 `Slice` + 1x1 `Conv` color mapで、exp023の高cost artifactより安く置換できる。

## Result

- base: `experiments/exp023_graph_surgery_exp016`
- target crop/resize tasks: `85`
- candidate rows: `170`
- generated candidates: `7`
- improved tasks: `0`
- local estimate: `6479.398825`
- delta: `+0.000000`
- gap to 6500: `20.601175`
- submission: no submit, 6500 threshold未達

## Interpretation

高cost側のcrop/resize taskは、単純な定数cropでは `output larger than input`、`no crop fits`、`no common crop` が多く、軽量 `Slice` 系だけでは説明できなかった。
validationを通った定数cropはtask135/task326などbaseline costがすでに極小のtaskで、candidate costがbaselineを上回った。

## Risks

- leakage risk: medium。train-fit候補をtest/arc-gen sample20で検証しているが、base自体はexp023のhigh-risk lookup upper boundを含む。
- overfitting risk: medium。sample20 passはprivate保証ではない。

## Decision

crop/resizeの次PDCAでは、定数cropを広げるよりも、same-shape global transformのartifact surgery、line/grid fillのsmall `Conv`/`Gather`、または既存artifact pruningを優先する。
