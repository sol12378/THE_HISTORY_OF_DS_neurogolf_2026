# exp041_task145_deeper_mul_chain

## Hypothesis

exp040でtask145は16passすべてfull-arc通過し、`Mul` node bypassが連続して効いた。task145だけに絞って最大32passまで深掘りすれば、探索時間を抑えながら安全な追加cost削減を得られる。

## Result

- base: `experiments/exp040_deeper_fullarc_safe_bypass`
- base local estimate: `6480.286117524449`
- target tasks: `145`
- safe ops: `And, Cast, Greater, Mul, Reshape, Slice, Squeeze, Transpose, Unsqueeze`
- max passes per task: `32`
- candidate rows: `3150`
- full-arc audited candidates: `261`
- accepted steps: `17`
- improved tasks: `1`
- improved task ids: `145`
- task145 cost: `62857 -> 61837`
- local estimate delta: `+0.016360414314510408`
- local estimate: `6480.302477938763`
- gap to 6500: `19.69752206123667`
- submission decision: `no_submit: below 6500 threshold`
- zip sanity: 400 files, names ok, sha256 `66b189b4cf0fe9f8ac0b0d2621d25e0afb43418a1071d0495de11a616401ab1b`

## Interpretation

task145の`Mul`連鎖はexp041でもさらに17 step削れた。candidateが途切れたため、単純な同一bypass深掘りはここで一段落。次はtask62の残り探索、またはtask145の構造を他taskへ一般化する。

## Risks

- leakage risk: medium。全arc-gen gate通過だが、base bundleにhigh-risk lookup artifactが残る。
- overfitting risk: medium。observed arc-genに基づくtask特化深掘りのためprivate distribution riskは残る。
- submit risk: 6500 threshold未達のため提出しない。
