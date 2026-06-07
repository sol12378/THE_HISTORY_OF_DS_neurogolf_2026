# exp038_fullarc_gated_noop_bypass

## Hypothesis

exp037の単純filterでは、exp036で落ちたtaskや既に改善済みtaskに対して「別の全arc-gen安全な中間候補」を取りこぼしている可能性がある。候補採用時点で全arc-gen gateを通すgreedy searchにすれば、安全寄りに追加改善を回収できる。

## Result

- base: `experiments/exp037_fullarc_filter_exp036`
- base local estimate: `6480.095137135056`
- top_k: `160`
- candidate rows: `6154`
- full-arc audited improved candidates: `459`
- accepted steps: `12`
- improved tasks: `4`
- improved task ids: `62, 145, 255, 268`
- local estimate delta: `+0.07586733661721468`
- local estimate: `6480.171004471673`
- gap to 6500: `19.82899552832714`
- submission decision: `no_submit: below 6500 threshold`
- zip sanity: 400 files, names ok, sha256 `1d25ac5a378cb890cf02e991c523e086efbfb85f423fa7c9c7c258fe529ef3a3`

## Validation

最終zip上で採用4 taskを再検証し、全arc-gen通過を確認した。

- task62: `267_pass_0_fail`
- task145: `267_pass_0_fail`
- task255: `265_pass_0_fail`
- task268: `266_pass_0_fail`

## Interpretation

full-arc gateをloop内へ入れる方向は正しい。exp037では落としたtask145/268からも、別の安全な逐次候補を回収できた。今後は `fullarc_candidate_audit.csv` から安全に通りやすいop/task patternを抽出し、候補生成順序とguardrailを改善する。

## Risks

- leakage risk: medium。全arc-genを通過しているが、baseにはhigh-risk lookup artifactが残る。
- overfitting risk: medium。sample20-only failureは抑えたが、private distribution riskは残る。
- submit risk: 6500 threshold未達のため提出しない。
