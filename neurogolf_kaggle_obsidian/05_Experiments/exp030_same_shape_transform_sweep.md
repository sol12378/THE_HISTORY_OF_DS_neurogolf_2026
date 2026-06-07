# exp030_same_shape_transform_sweep

## Hypothesis

same-shape系とlookup由来taskの一部は、trainだけから推定できる軽量な `Gather`/`Transpose` global transform、1x1 `Conv` color map、またはidentityでexp023 artifactより安く置換できる。

## Result

- base: `experiments/exp023_graph_surgery_exp016`
- target tasks: `140`
- candidate rows: `420`
- generated candidates: `0`
- improved tasks: `0`
- local estimate: `6479.398825`
- delta: `+0.000000`
- gap to 6500: `20.601175`
- submission: no submit, 6500 threshold未達

## Interpretation

対象taskは単純なglobal transform、consistent color map、identityではtrain-fitしなかった。
`task255` も `no global transform fits` / `conflicting color map` / `not identity` で、より構造的なsame-shape ruleが必要。

## Risks

- leakage risk: low-medium。train-fitの説明可能候補だけを試したが、baseはhigh-risk lookup upper boundを含む。
- overfitting risk: medium。sample20 validationはprivate-like holdoutではない。

## Decision

次PDCAはline/grid fillまたはpoint-to-line patternへ移し、small `Conv`/`Gather` と明示的cost gateだけを使う。
