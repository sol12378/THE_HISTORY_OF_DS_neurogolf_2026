# exp120_task366_intermediate_output_extraction

## Hypothesis

task366既存artifact内部に、正しいfinal-shape tensorが早い段階で存在するなら、そのtensorをoutput化してsuffixを切れる。

## Result

- campaign index: `22`
- baseline cost: `99849`
- all intermediate tensors: `722`
- final-shape candidate tensors: `6`
- status counts: `rejected=6`
- validation counts: `not_run=6`
- accepted tasks: `[]`
- local delta: `0.000000`

Top shape counts:

- `1x1x30x30`: `311`
- `1x1x1x1`: `229`
- `1x1x1x30`: `41`
- `1x1x30x1`: `41`
- `1x1x59x59`: `39`
- `1x10x30x30`: `6`

## Interpretation

task366 artifactは大きなmask生成に支配されている。final-shape candidateは6件あるが、いずれもelem type mismatchでoutput化できない。既存artifactの単純suffix cutでは改善できない。

次にtask366を続けるなら、mask-to-float suffixを最小化して切れるか、またはobject-marker copy ruleのfresh representationを改めて設計する必要がある。ただし過去のfull-grid primitive / sparse update probeではcostが高いため、慎重に進める。

## Risk

- leakage risk: low。graph-only extraction。
- overfitting risk: low-to-medium。
