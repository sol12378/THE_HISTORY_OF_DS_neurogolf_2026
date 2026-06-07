# exp040_deeper_fullarc_safe_bypass

## Hypothesis

exp039で採用されたfull-arc-safe bypass chainは、まだ同じop/task系列に未削除のno-op nodeが残っている。baseをexp039に切り替えて最大16passまで深掘りすれば、広い探索を増やさずに安全な追加cost削減を得られる。

## Result

- base: `experiments/exp039_deep_fullarc_safe_bypass`
- base local estimate: `6480.235261967659`
- target tasks: `55, 58, 62, 74, 96, 138, 145, 148, 233, 255, 268, 289, 392, 398`
- safe ops: `And, Cast, Greater, Mul, Reshape, Slice, Squeeze, Transpose, Unsqueeze`
- max passes per task: `16`
- candidate rows: `6248`
- full-arc audited candidates: `714`
- accepted steps: `38`
- improved tasks: `4`
- improved task ids: `62, 145, 255, 268`
- local estimate delta: `+0.05085555678929943`
- local estimate: `6480.286117524449`
- gap to 6500: `19.713882475551145`
- submission decision: `no_submit: below 6500 threshold`
- zip sanity: 400 files, names ok, sha256 `6532b5b54d5bdbede7d62bd73aa2146bdc86bbc88f21afb352293c8211b49df7`

## Interpretation

同じsafe-op系列はまだ伸びる。大きなgainはtask62とtask145で、task255とtask268は小さいが安定して積み増せた。次はtask62/145にさらに絞った深掘り、またはfullarc auditから`Mul`連鎖削除の一般化を試す。

## Risks

- leakage risk: medium。全arc-gen gate通過だが、base bundleにhigh-risk lookup artifactが残る。
- overfitting risk: medium。observed arc-genに基づくsafe pattern深掘りのため、private distribution riskは残る。
- submit risk: 6500 threshold未達のため提出しない。
