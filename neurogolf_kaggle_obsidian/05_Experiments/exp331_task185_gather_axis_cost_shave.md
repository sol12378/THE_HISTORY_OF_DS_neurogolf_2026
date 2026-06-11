# exp331_task185_gather_axis_cost_shave

## Hypothesis

exp330 の `Tile + GatherElements` dynamic index tensor を ONNX `Gather(axis=2/3)` に置き換えると、task185 の full-arc correctness を維持したまま baseline cost `59584` 未満へ落とせる。

## Plan

- exp329/330 の dynamic bg detector + axis selector + homogeneous 2x2 core は維持する。
- `row_vec` / `col_vec` を rank-1 dynamic index として作り、`Gather(axis=2)` と `Gather(axis=3)` で 4x4 lattice を抽出する。
- full-arc validation と official `score_network` cost を確認する。

## Result

- variant: `gather_axis_float`
- validation: `267_pass_0_fail`
- static / score / runtime: `ok`
- cost: `59584 -> 48889`
- local delta: `+0.19783465935382516`
- expected Public LB if calibrated: `6009.157834659354`
- submission: Kaggle ref `53553605`, status `COMPLETE`, Public LB `6009.15`

## Decision

提出価値あり。exp297 best bundle に `task185.onnx` だけを差し替えた 400-file zip を作成し、Kaggle に提出した。Public LB `6009.15` で expected `6009.157834659354` と丸め範囲で一致したため、新しい best として採用する。

## Interpretation

`Gather(axis=2/3)` は exp330 の cost wall の主因だった Tile index memory を削る有効な表現だった。task185 は solved-rule lowering として初めて score-direct candidate になったため、同様の dynamic index 縮小を他の GridSample/Gather queue でも再利用する価値がある。

## Risk

- leakage risk: low。input-only detector/selector と deterministic lattice extraction で、public output lookup は使っていない。
- overfitting risk: medium-low。task-specific geometry だが full-arc gate を通過しており、public artifact raw 依存ではない。

## Next

task185 は採用済み。次は task251 mask+color1 または task037 sparse diagonal lowering に着手する。
