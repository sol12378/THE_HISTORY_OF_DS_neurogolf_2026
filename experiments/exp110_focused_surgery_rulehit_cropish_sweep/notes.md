# exp110_focused_surgery_rulehit_cropish_sweep

## Hypothesis

exp109 showed that focused bypass surgery on a rule-hit task can produce a safe micro-delta. Sweep the same conservative surgery over top P0 cropish tasks plus selected rule-hit tasks, using exp109 as the current base.

## Result

- targets: `35`
- generated candidates: `1299`
- status counts: `{'rejected': 1240, 'no_cost_gain': 37, 'improved': 22}`
- accepted tasks: `[184, 187, 207, 263, 316, 394]`
- local delta: `0.086887`
- new local estimate: `6282.922190`

## Interpretation

If accepted tasks are found, focused surgery should be kept as a calibration lane. If not, exp109 may be a narrow task048-specific cleanup and #13 should move to fused lowering/subgraph extraction.

## Risk

- leakage risk: low。
- overfitting risk: low-to-medium。acceptedはfull validation済みだがLB calibration対象。
