# exp036_noop_bypass_and_prune

## Hypothesis

exp035後のbundleにも、公式sanitizeで消せる未使用initializerや、実質no-opになっているshape/cast/arithmetic nodeが残っている可能性がある。`Cast`, `Reshape`, `Squeeze`, `Unsqueeze`, `Transpose`, `Pad`, `Slice`, `Gather`, `Add`, `Mul` などを片入力へbypassし、sample20 validationで守れば追加のcost削減を拾える。

## Result

- base: `experiments/exp035_greedy_logic_surgery_composition`
- base local estimate: `6479.662741382307`
- top_k: `160`
- candidate rows: `7777`
- accepted steps: `31`
- improved tasks: `16`
- sample20 local estimate: `6480.195302061119`
- delta vs exp035: `+0.5325606788123327`
- gap to 6500: `19.804697938880963`
- submission decision: `no_submit: below 6500 threshold`

## Full Arc-Gen Audit

全arc-genで追加検証したところ、以下4 taskがsample20-only改善だった。

- rejected: `80`, `145`, `187`, `268`
- kept: `55`, `58`, `62`, `74`, `96`, `138`, `148`, `233`, `255`, `289`, `392`, `398`

## Interpretation

広いno-op bypassは強い探索方向だった。特にtask289の `Cast` bypass、task74の `Transpose` bypass、task255の `Cast/Gather/Unsqueeze` 系削減が大きい。一方で、sample20だけで採用すると後半arc-genで破綻する候補が混ざるため、今後の広いgraph surgeryはfull-arc acceptanceをloop内に入れるべき。

## Risks

- leakage risk: medium。sample20採用段階ではbaseのlookup依存とvalidation過学習が残る。
- overfitting risk: high。全arc-gen監査で実際に4 taskのsample20-only failureを確認した。
- submit risk: 6500未達かつ一部full-arc failureのため、このbundleは提出しない。
