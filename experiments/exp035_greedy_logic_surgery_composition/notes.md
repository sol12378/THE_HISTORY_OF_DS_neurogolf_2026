# exp035_greedy_logic_surgery_composition

## Hypothesis

exp033/034の単発graph surgeryは、同じtaskに対して逐次合成できる。改善候補を1つ採用した後に再度候補生成すれば、さらに冗長nodeを削れる可能性がある。

## Result

- base: `experiments\exp034_redundant_logic_surgery_sweep`
- top_k: `120`
- max passes per task: `4`
- candidate rows: `2668`
- accepted steps: `16`
- improved tasks: `7`
- local estimate: `6479.662741`
- delta: `0.078233`
- gap to 6500: `20.337259`
- submission decision: `no_submit: below 6500 threshold`

## Risks

- leakage risk: medium: greedy graph surgery is accepted only after sample20 validation, but base includes high-risk lookup artifacts.
- overfitting risk: medium-high: sequential edits can compound sample-specific risk without full private-like validation.
