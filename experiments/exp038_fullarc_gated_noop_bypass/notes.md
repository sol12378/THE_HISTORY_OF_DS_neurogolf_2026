# exp038_fullarc_gated_noop_bypass

## Hypothesis

exp037をbaseにして、候補採用時点で全arc-gen gateを通せば、exp036で見つけた広いno-op bypassの追加余地を安全寄りに回収できる。

## Result

- base: `experiments\exp037_fullarc_filter_exp036`
- candidate rows: `6154`
- full-arc audited candidates: `459`
- accepted steps: `12`
- improved tasks: `4`
- local estimate: `6480.171004`
- delta: `0.075867`
- gap to 6500: `19.828996`
- submission decision: `no_submit: below 6500 threshold`

## Risks

- leakage risk: medium: edits pass all available arc-gen, but base still includes lookup artifacts.
- overfitting risk: medium: full-arc gate is stronger than sample20, but private distribution risk remains.
