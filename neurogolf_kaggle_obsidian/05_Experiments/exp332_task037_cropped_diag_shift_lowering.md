# exp332_task037_cropped_diag_shift_lowering

## Hypothesis

task037 の bounded diagonal shift rule は exp327 で full-arc valid だったが、30x30 full-grid intermediate により cost `6633317` で不採用だった。入力を working area の 10x10 に crop してから同じ rule を実行し、最後に 30x30 へ Pad すれば cost が大きく下がる。

## Result

- base: current best lineage `exp331`
- base task037 cost: `63726`
- candidate: `cropped_diag_shift_d5`
- validation: `266_pass_0_fail`
- candidate cost: `721329`
- candidate status: `no_cost_gain`
- local delta: `0.0`
- submission: no submit

## Interpretation

10x10 crop は exp327 の `6633317` から `721329` まで cost を下げたが、baseline `63726` には届かなかった。したがって task037 の dense shift-stack は cropped でも不十分。task037 を続けるなら full-grid/dense shift ではなく、true per-diagonal sparse representation が必要。

## Decision

No submit。短期 score-direct では task251 mask+color1 へ pivot する。task037 は per-diagonal sparse 設計が見えたときのみ再開する。

## Risk

- leakage risk: low。input-only diagonal ray rule で public output lookup は使っていない。
- overfitting risk: medium。task-specific geometry は full-arc valid だが、表現が task037 に強く依存する。

## Next

task251 の closed-component mask + cheap color1/recolor minimal probe を実行する。
