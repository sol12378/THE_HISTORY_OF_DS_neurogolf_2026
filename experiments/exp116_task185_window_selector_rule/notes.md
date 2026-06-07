# exp116_task185_window_selector_rule

## Hypothesis

task185のlattice位置はraw coordinate tableではなく、grid line上の4連続windowをscoreして選べる。

## Result

- pass: `267/267`
- fail: `0`
- local delta: `0.000000`

## Interpretation

full passなら、次はONNXでwindow scoringを低cost化する。partialなら失敗例からtie-breakerを追加する。

## Risk

- leakage risk: low。
- overfitting risk: medium-low。window scoringなら許容、raw position table化は避ける。
