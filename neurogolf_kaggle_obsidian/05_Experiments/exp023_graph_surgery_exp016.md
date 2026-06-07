# exp023_graph_surgery_exp016

## Hypothesis

exp016 の signature lookup bundle には、未使用initializerが残っており、低リスクなgraph surgeryでlocal estimateを少し改善できる可能性がある。

## Result

- base: `exp016_top100_rewrite_campaign`
- baseline local estimate: `6479.383086`
- new local estimate: `6479.398825`
- delta: `0.015738`
- gap to 6500: `20.601175`
- improved task count: `195`
- changed variant: `surgery_prune_unused_initializers`

## Interpretation

exp016の `signature_scatternd_lookup` 系には未使用initializerが残っていた。点数改善は小さいが、今後の候補生成では最初から余計なinitializerを出さないよう `phase1_rewrite_utils.py` も修正した。

## Risks

- leakage risk: high。base は exp016 の signature lookup を含む。
- overfitting risk: high。validation は sample20 であり、full arc-genではない。

