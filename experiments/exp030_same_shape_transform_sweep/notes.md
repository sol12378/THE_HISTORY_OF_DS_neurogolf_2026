# exp030_same_shape_transform_sweep

## Hypothesis

same-shape系とlookup由来taskの一部は、trainだけから推定できる軽量な `Gather`/`Transpose` global transform、1x1 `Conv` color map、またはidentityでexp023 artifactより安く置換できる。

## Result

- base: `experiments\exp023_graph_surgery_exp016`
- target tasks: `140`
- candidate rows: `420`
- generated candidates: `0`
- improved tasks: `0`
- local estimate: `6479.398825`
- delta: `0.000000`
- gap to 6500: `20.601175`
- submission decision: `no_submit: below 6500 threshold`

## Interpretation

full-grid loweringやlarge `ScatterND` は使わず、cost guardrail上で軽い候補だけを試した。
改善が出ても6500未満なら提出しない。6500以上ならfull validationとrisk review後に提出候補にする。

## Risks

- leakage risk: low-medium: train-fitted global transforms are interpretable, but base includes high-risk lookup artifacts.
- overfitting risk: medium: sample20 validation is not private-like family holdout.
