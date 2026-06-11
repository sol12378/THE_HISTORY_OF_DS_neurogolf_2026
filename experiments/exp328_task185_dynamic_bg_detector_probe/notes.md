# exp328_task185_dynamic_bg_detector_probe

## 目的

task185 の残課題である dynamic background detector を、入力の非ゼロ最頻色として検証し、ONNX subgraph の official cost を測る。

## 結果

- detector validation: `267_pass_0_fail`
- detector cost: `143`
- memory/params: `132` / `11`
- bg_hist: `{1: 32, 2: 23, 3: 30, 4: 25, 5: 26, 6: 26, 7: 34, 8: 37, 9: 34}`

## 判断

Dynamic bg detector is ready for the next task185 candidate.

## Submission

`no_submit: detector subgraph probe only`

## Risk

- leakage risk: low: input-only modal-color detector; no output lookup or public feedback.
- overfitting risk: low-to-medium: modal-bg assumption is task-specific but validated over all local examples.
