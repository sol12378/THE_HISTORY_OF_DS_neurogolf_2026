# exp036_noop_bypass_and_prune

## Hypothesis

exp035後のbundleにも、公式sanitizeで消せる未使用initializerや、実質no-opになっているshape/cast/arithmetic nodeが残っている可能性がある。

## Result

- base: `experiments\exp035_greedy_logic_surgery_composition`
- top_k: `160`
- max passes per task: `3`
- candidate rows: `7777`
- accepted steps: `31`
- improved tasks: `16`
- local estimate: `6480.195302`
- delta: `0.532561`
- gap to 6500: `19.804698`
- submission decision: `no_submit: below 6500 threshold`

## Risks

- leakage risk: medium: candidates are validated on sample20 and base contains high-risk lookup artifacts.
- overfitting risk: medium-high: broad no-op bypass search can overfit validation examples.
