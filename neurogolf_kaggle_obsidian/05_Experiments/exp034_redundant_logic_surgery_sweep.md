# exp034_redundant_logic_surgery_sweep

## Hypothesis

exp033の `And` surgery以外にも、`Or`、comparison、`Where` nodeの出力を片方の入力に置換してもsemanticsが変わらない冗長箇所がある。

## Result

- base: `experiments/exp033_redundant_and_surgery_sweep`
- top_k: `160`
- candidate rows: `1234`
- improved tasks: `6`
- improved task ids: `80, 86, 138, 148, 187, 289`
- local estimate: `6479.584508`
- delta: `+0.067392`
- gap to 6500: `20.415492`
- submission: no submit, 6500 threshold未達

## Top Improvements

- task138: cost `48467 -> 47567`
- task289: cost `24972 -> 24509`
- task148: cost `22029 -> 21741`
- task187: cost `103513 -> 102613`
- task86: cost `25713 -> 25569`
- task80: cost `41282 -> 41182`

## Interpretation

`And`以外にも冗長logic nodeは存在し、sample20 validationとofficial-like scoreを通せば安全に小改善を積める。
6500まではまだ `20.415492` あるため提出しない。

## Risks

- leakage risk: medium。graph surgeryはsample20で意味保持を確認しているが、baseはhigh-risk lookup upper boundを含む。
- overfitting risk: medium。sample20 validationはprivate-like holdoutではない。

## Decision

exp034 bundleを新しいlocal upper boundとして採用する。6500未達のため提出なし。
