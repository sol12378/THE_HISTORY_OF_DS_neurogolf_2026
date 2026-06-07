# exp119_task365_intermediate_output_extraction

## Hypothesis

task365既存artifact内部に、正しいfinal-shape tensorが早い段階で存在するなら、そのtensorをoutput化してsuffixを切れる。

## Result

- campaign index: `21`
- baseline cost: `58814`
- all intermediate tensors: `235`
- final-shape candidate tensors: `4`
- status counts: `rejected=4`
- validation counts: `not_run=3`, `0_pass_1_fail=1`
- accepted tasks: `[]`
- local delta: `0.000000`

## Interpretation

task365の既存artifactにも単純なsuffix cut余地はない。final-shape候補のうち3件はelem type mismatch、1件はexample 0 mismatchで落ちた。

次はfresh selector loweringのcost floorを測るか、より高gainのtask366へpivotする。task365は6x6 `Slice+Pad` floorが既に `1461` で、object selection前に600を超えるため、score-producingまでの難度は高い。

## Risk

- leakage risk: low。graph-only extraction。
- overfitting risk: low-to-medium。
