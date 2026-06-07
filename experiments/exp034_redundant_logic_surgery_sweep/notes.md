# exp034_redundant_logic_surgery_sweep

## Hypothesis

exp033の `And` surgery以外にも、`Or`、comparison、`Where` nodeの出力を片方の入力に置換してもsemanticsが変わらない冗長箇所がある。

## Result

- base: `experiments\exp033_redundant_and_surgery_sweep`
- top_k: `160`
- candidate rows: `1234`
- improved tasks: `6`
- local estimate: `6479.584508`
- delta: `0.067392`
- gap to 6500: `20.415492`
- submission decision: `no_submit: below 6500 threshold`

## Risks

- leakage risk: medium: graph surgery is accepted only after sample20 validation, but base includes high-risk lookup artifacts.
- overfitting risk: medium: validation is train/test/arc-gen sample20, not full private-like holdout.
