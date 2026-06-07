# exp037_fullarc_filter_exp036

## Hypothesis

exp036のsample20改善から、全arc-genを通過した候補だけを残せば、過学習候補を落としつつ堅いlocal estimateを更新できる。

## Result

- base: `experiments/exp035_greedy_logic_surgery_composition`
- source: `experiments/exp036_noop_bypass_and_prune`
- base local estimate: `6479.662741382307`
- source sample20 local estimate: `6480.195302061119`
- source selected tasks: `16`
- kept after full arc-gen: `12`
- rejected after full arc-gen: `4`
- kept tasks: `55, 58, 62, 74, 96, 138, 148, 233, 255, 289, 392, 398`
- rejected tasks: `80, 145, 187, 268`
- local estimate: `6480.095137135056`
- delta vs exp035: `+0.4323957527484481`
- gap to 6500: `19.904862864944334`
- submission decision: `no_submit: below 6500 threshold`
- zip sanity: 400 files, names ok, sha256 `e932add1fc10763a55ff4050fff2534cc1c7fe6659781f0150cbc7a16535aef9`

## Interpretation

exp037を新しい堅めのbest local upper boundとして採用する。exp036の広い探索は有効だが、acceptance gateをsample20に置くとsample後半で破綻する候補を拾う。今後はfull-arc acceptanceを標準にし、候補数を絞るためにtask/node優先度を先に付ける。

## Risks

- leakage risk: medium。全arc-genを通しているが、baseにはhigh-risk lookup artifactが残る。
- overfitting risk: medium。観測可能なarc-gen failureは除去したが、private distribution riskは残る。
- submit risk: 6500 threshold未達のため提出しない。

## Next Actions

- exp036のrejected taskを分析し、危険なbypass patternをguardrail化する。
- full-arc acceptanceをloop内に入れたexp038を走らせる。
- no-op bypass以外の低cost lowering、特にline/grid fillとpoint-to-line synthesisを継続する。
