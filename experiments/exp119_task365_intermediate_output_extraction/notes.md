# exp119_task365_intermediate_output_extraction

## Hypothesis

task365既存artifact内部に、正しいfinal-shape tensorが早い段階で存在するなら、そのtensorをoutput化してsuffixを切れる。

## Result

- candidate tensors: `4`
- status counts: `{'rejected': 4}`
- validation counts: `{'not_run': 3, '0_pass_1_fail': 1}`
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.936157`

## Interpretation

suffix extractionで改善が出れば、task365をsubmit-safe post-pass候補へ入れる。出なければfresh selector loweringか別taskへ移る。

## Risk

- leakage risk: low。
- overfitting risk: low-to-medium。
