# exp107_task185_logic_bypass_surgery

## Hypothesis

task185 current artifact has many generated logic/mask nodes (`And`, `Equal`, `Not`, `Where`). Some guards may be redundant under official validation, and bypassing a node output to one input may reduce official cost while preserving correctness.

## Result

- campaign #: `9`
- base local estimate: `6282.812218`
- task: `185`
- baseline cost: `59584`
- generated candidates: `240`
- full validation pass candidates: `39`
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.812218`
- submission decision: `no_submit`

## Interpretation

39 candidates passed full validation, but all stayed at cost `59584`. Local logic bypass does not reduce the official score for task185 after sanitize/score. task185 needs a larger fused replacement/lowering for the 4x4 lattice -> 3x3 homogeneous 2x2 rule, not local guard deletion.

## Risk

- leakage risk: low。graph surgery only。
- overfitting risk: low-to-medium。full validation pass候補はあったがno gain。
