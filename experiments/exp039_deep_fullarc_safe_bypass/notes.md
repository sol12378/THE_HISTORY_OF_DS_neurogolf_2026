# exp039_deep_fullarc_safe_bypass

## Hypothesis

exp037/038で全arc-gen通過実績があるtaskとopに絞って深いpassを回せば、広い探索より安全かつ高速に追加cost削減を得られる。

## Result

- base: `experiments\exp038_fullarc_gated_noop_bypass`
- task count: `14`
- safe ops: `And, Cast, Greater, Mul, Reshape, Slice, Squeeze, Transpose, Unsqueeze`
- candidate rows: `4878`
- full-arc audited candidates: `776`
- accepted steps: `32`
- improved tasks: `4`
- local estimate: `6480.235262`
- delta: `0.064257`
- gap to 6500: `19.764738`
- submission decision: `no_submit: below 6500 threshold`

## Risks

- leakage risk: medium: full-arc gated edits, but base still includes high-risk lookup artifacts.
- overfitting risk: medium: targeted by observed safe patterns; private distribution risk remains.
