# exp120_task366_intermediate_output_extraction

## Hypothesis

task366既存artifact内部に、正しいfinal-shape tensorが早い段階で存在するなら、そのtensorをoutput化してsuffixを切れる。

## Result

- candidate tensors: `6`
- status counts: `{'rejected': 6}`
- validation counts: `{'not_run': 6}`
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.936157`

## Interpretation

suffix extractionで改善が出れば、task366をsubmit-safe post-pass候補へ入れる。出なければfresh low-cost representationが必要。

## Risk

- leakage risk: low。
- overfitting risk: low-to-medium。
