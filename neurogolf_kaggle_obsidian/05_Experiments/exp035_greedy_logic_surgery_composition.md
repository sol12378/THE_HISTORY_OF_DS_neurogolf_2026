# exp035_greedy_logic_surgery_composition

## Hypothesis

exp033/034で有効だった単発graph surgeryは、同じtaskに対して逐次合成できる可能性がある。1つの冗長nodeを削った後に再度候補生成すれば、追加の冗長 `And` / `Or` / comparison / `Where` nodeを発見できる。

## Setup

- base: `experiments/exp034_redundant_logic_surgery_sweep`
- accounting: exp032、exp033、exp034の改善を順番にoverlayしてbase local estimateを算出
- top_k: `120`
- max passes per task: `4`
- arc-gen sample: `20`
- candidate ops: `And`, `Or`, `Greater`, `Less`, `LessOrEqual`, `GreaterOrEqual`, `Equal`, `Where`

## Result

- base local estimate: `6479.584508334562`
- candidate rows: `2668`
- accepted steps: `16`
- improved tasks: `7`
- improved task ids: `80, 138, 145, 187, 268, 289, 376`
- local estimate delta: `+0.0782330477454316`
- local estimate: `6479.662741382307`
- gap to 6500: `20.337258617692896`
- submission decision: `no_submit: below 6500 threshold`
- zip sanity: `400` files, names ok, sha256 `c8ef3402426ef199b620547dfafed6ba8e11e9b3b7fc6b7c0162ca398ef87d87`

## Interpretation

逐次合成は有効だった。特にtask187は3段、task80とtask268は4段まで小さな削減を積めた。一方、増分は `+0.0782` に留まり、6500までの残り `20.3373` を埋める主戦力にはまだ足りない。

初回実行ではbase accountingがexp034以前の改善を一部しか反映せず、base local estimateが低く表示された。修正版ではexp032、exp033、exp034のselected改善を明示的にoverlayし、`6479.584508334562` を正しいbaseとして再計算した。

## Risks

- leakage risk: medium。sample20 validationでのみ採用しており、baseにはhigh-risk lookup artifactが含まれる。
- overfitting risk: medium-high。逐次編集はsample固有の偶然を積み上げる可能性がある。
- submit risk: 6500 threshold未達のため提出しない。

## Next Actions

- exp035 bundle上でunused initializer pruningを再度走らせる。
- `And` / `Or` / comparison / `Where` 以外の代数的graph surgery候補を増やす。
- 6500 gapを埋めるにはgraph surgeryだけでなく、line/grid fillやpoint-to-line synthesisの低cost loweringを並行する。
