# exp331_task185_gather_axis_cost_shave

## 目的

exp330 の task185 full candidate は `267_pass_0_fail` だが baseline より cost が `+200` 高かった。原因候補である `Tile + GatherElements` の大きな index tensor を、ONNX `Gather(axis=2/3)` に置き換えて cost shave する。

## 仮説

`row_vec` / `col_vec` を rank-1 dynamic index として使えば、`row_idx [1,10,4,30]` と `col_idx [1,10,4,4]` を materialize せずに同じ 4x4 lattice を抽出できる。

## 結果

- baseline_cost: `59584`
- rows: `{'gather_axis_float': {'validation': '267_pass_0_fail', 'status': 'improved', 'cost': 48889, 'delta': 0.19783465935382516}}`
- best_row: `gather_axis_float`
- best_cost: `48889`
- decision: `Submit only if bundled delta exceeds the public display threshold.`
- submission_decision: `submit_after_review`
- submission: Kaggle ref `53553605`, status `COMPLETE`, Public LB `6009.15`.

## 解釈

`Gather(axis=2/3)` は full-arc valid で、exp330 の `Tile + GatherElements` より cost を大きく削れた。task185 は `59584 -> 48889`、local delta `+0.19783465935382516`。exp297 best bundle に task185 だけ差し替えて提出した結果、Public LB は `6009.15` となり、期待値 `6009.157834659354` と丸め範囲で一致した。新しい best として採用する。

## Risk

- leakage risk: low: input-only detector/selector and deterministic lattice extraction; no output lookup or public-score-driven constants.
- overfitting risk: medium-low: task-specific geometry, but all examples must pass full-arc gate and no public repair source is used.
