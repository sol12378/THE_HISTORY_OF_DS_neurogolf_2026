# exp111_focused_surgery_second_pass

## Hypothesis

After exp109/exp110 accepted focused bypass edits, a second pass may expose additional redundant guards/casts. This tests whether focused surgery is a compoundable post-pass or mostly a one-shot cleanup.

## Result

- targets: `35`
- generated candidates: `1417`
- status counts: `{'rejected': 1354, 'no_cost_gain': 41, 'improved': 22}`
- accepted tasks: `[184, 187, 263, 316, 394]`
- local delta: `0.013967`
- new local estimate: `6282.936157`

## Decision

If second-pass delta is small or zero, focused surgery remains a useful but shallow post-pass. If it compounds, use greedy multi-pass surgery before LB calibration.

## Risk

- leakage risk: low。
- overfitting risk: low-to-medium。
