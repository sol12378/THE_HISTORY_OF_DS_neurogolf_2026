# exp332_task037_cropped_diag_shift_lowering

## 目的

exp327 の full-grid 30x30 bounded shift stack は正しいが cost `6633317` で不採用だった。task037 の実 working area は 10x10 なので、最初に crop してから同じ rule を実行し、最後に 30x30 へ Pad すれば memory cost が下がるか確認する。

## 結果

- base cost: `63726`
- candidate validation: `266_pass_0_fail`
- candidate status: `no_cost_gain`
- candidate cost: `721329`
- local delta: `0.0`

## 判断

`no_submit`: full-arc は通ったが、cost が baseline を大きく超えたため採用しない。

## 解釈

10x10 crop により exp327 の `6633317` から `721329` までは下がったが、baseline `63726` には届かない。dense shift-stack は cropped でもまだ重い。task037 を続けるなら true per-diagonal sparse representation が必要で、短期 score-direct では task251 mask+color1 へ pivot する方が妥当。

## Risk

- leakage risk: low: explicit input-only diagonal ray rule; no output lookup or public feedback.
- overfitting risk: medium: task-specific geometry is full-arc validated locally, but cost expression may not generalize beyond this task family.
