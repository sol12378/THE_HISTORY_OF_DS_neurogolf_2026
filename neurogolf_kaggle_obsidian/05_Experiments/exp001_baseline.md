# exp001_baseline

---
tags: [experiment]
status: submitted_complete
exp_id: exp001_baseline
model: baseline
cv: task087 local all pass
lb: 14.50
---

## Hypothesis

A minimal solver establishes the data loading, validation, and submission pipeline.

## Changes

- 2026-06-05: `docs/neurogolf_first_submission_plan.md` を確認。
- 2026-06-05: `.venv` に `kaggle`, `onnx`, `onnxruntime`, `onnx-tool`, `numpy` を導入。
- 2026-06-05: `nvidia-smi` でGPU driverを確認。RTX 2080系 8GB VRAM、CUDA driver 13.1。
- 2026-06-05: Kaggle API認証通過。`data/raw` は触らず `data/external/neurogolf-2026` にデータ取得。
- 2026-06-05: 公式Evaluation/Data/Constraints/May 4 updateを確認。MACsはcost対象外。
- 2026-06-05: `task087` を選び、3x3 180度回転を `Gather` 2段の手書きONNXで実装。
- 2026-06-05: Kaggle APIで `submission.zip` を提出。ref `53383536`。

## Results

| Metric | Value |
|---|---:|
| CV | task087 local all pass |
| LB | 14.50 |
| ONNX env | OK |
| Kaggle API | OK |
| ARC-AGI pass/fail | 5 / 0 |
| ARC-GEN pass/fail | 261 / 0 |
| Memory bytes | 36000 |
| Params | 60 |
| Estimated points | 14.507 |
| ONNX file size | 299 bytes |

## Decision

submitted_complete

## Notes

初回計画はGPU学習ではなく、CPUでONNXの正しさ検証とコスト計測を行う方針。GPUはdriverから見えているが、今回の検証はCPUで完結した。

作成物:
- `experiments/exp001_baseline/task087.onnx`
- `experiments/exp001_baseline/submission.zip`
- `experiments/exp001_baseline/task087_result.json`

Kaggle submit済み。ref `53383536`、status `COMPLETE`、publicScore `14.50`。

Leakage risk: public examplesに対する既知taskの手書きONNXのため低いが、private benchmarkで同じ規則が成立する保証はない。

Overfitting risk: task087に固定した3x3 rot180であり、中。public LB最適化だけを追わない。
