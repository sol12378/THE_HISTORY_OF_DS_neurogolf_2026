# exp033_redundant_and_surgery_sweep

## Hypothesis

exp032で有効だった `And(mask, already_masked)` 型の冗長nodeは、他の高cost artifactにも存在する。各 `And` nodeの出力を片方の入力に置換してもvalidationが通る場合、1 node分のmemory costを削減できる。

## Result

- base: `experiments\exp032_task187_component_lowering`
- top_k: `120`
- candidate rows: `326`
- improved tasks: `8`
- local estimate: `6479.517116`
- delta: `0.109709`
- gap to 6500: `20.482884`
- submission decision: `no_submit: below 6500 threshold`

## Interpretation

改善候補はsample20 validationとofficial-like scoreを通したものだけ採用する。
6500未満なら提出しない。

## Risks

- leakage risk: medium: graph surgery preserves candidate semantics under sample20 validation, but base includes high-risk lookup artifacts.
- overfitting risk: medium: node replacement is validated on train/test/arc-gen sample20, not full private-like holdout.
