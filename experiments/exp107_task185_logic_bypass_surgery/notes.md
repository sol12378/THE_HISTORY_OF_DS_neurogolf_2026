# exp107_task185_logic_bypass_surgery

## Hypothesis

task185 current artifact has many logic/mask nodes (`And`, `Equal`, `Not`, `Where`). Some generated guards may be redundant under the official examples. Bypassing a redundant node output to one of its inputs may reduce cost while preserving full validation.

## Result

- generated candidates: `240`
- status counts: `{'rejected': 201, 'no_cost_gain': 39}`
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.812218`

## Interpretation

This is a score-producing graph-surgery attempt on a rule-hit task. If no accepted candidate appears, task185 needs a fused replacement/lowering rather than local logic bypass.

## Risk

- leakage risk: low。
- overfitting risk: low-to-medium。acceptedが出た場合はsingle-task LB calibration候補。
