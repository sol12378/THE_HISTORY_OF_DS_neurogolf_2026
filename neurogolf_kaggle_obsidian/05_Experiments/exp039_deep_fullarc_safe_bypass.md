# exp039_deep_fullarc_safe_bypass

## Hypothesis

exp037/038で全arc-gen通過実績があるtaskとopに絞って深いpassを回せば、広い探索より安全かつ高速に追加cost削減を得られる。

## Result

- base: `experiments/exp038_fullarc_gated_noop_bypass`
- base local estimate: `6480.171004471673`
- target tasks: `55, 58, 62, 74, 96, 138, 145, 148, 233, 255, 268, 289, 392, 398`
- safe ops: `And, Cast, Greater, Mul, Reshape, Slice, Squeeze, Transpose, Unsqueeze`
- max passes per task: `8`
- candidate rows: `4878`
- full-arc audited candidates: `776`
- accepted steps: `32`
- improved tasks: `4`
- improved task ids: `62, 145, 255, 268`
- local estimate delta: `+0.0642574959861335`
- local estimate: `6480.235261967659`
- gap to 6500: `19.764738032340574`
- submission decision: `no_submit: below 6500 threshold`
- zip sanity: 400 files, names ok, sha256 `e75d2f11182df2c8051bee171932ab21d945030eec977e7a1978bdee44de78ef`

## Validation

最終zip上で採用4 taskを再検証し、全arc-gen通過を確認した。

- task62: `267_pass_0_fail`
- task145: `267_pass_0_fail`
- task255: `265_pass_0_fail`
- task268: `266_pass_0_fail`

## Interpretation

safe-op deepeningはまだ効く。task62は`Mul/Transpose`の交互削減で大きく伸び、task145は`Mul`系列、task255は`Reshape`系列、task268は`Greater/And`系列で小さく積めた。今後は同じ4 taskでpassをさらに延ばすか、safe patternを他taskへ一般化する。

## Risks

- leakage risk: medium。全arc-genを通過しているが、baseにはhigh-risk lookup artifactが残る。
- overfitting risk: medium。観測可能なarc-gen failureは抑えているが、private distribution riskは残る。
- submit risk: 6500 threshold未達のため提出しない。
