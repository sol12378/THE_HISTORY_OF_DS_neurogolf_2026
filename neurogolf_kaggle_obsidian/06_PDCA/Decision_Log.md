# Decision Log

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-06-05 | exp001はCPU ONNX検証を優先し、GPU学習は次フェーズ以降に回す。 | 初回提出計画はパイプライン一周が目的で、`onnxruntime` CPU検証と公式cost確認で十分。 | `task087.onnx` と単一task `submission.zip` を作成し、Kaggle submit ref `53383536` / public LB `14.50` を記録。 |
| 2026-06-05 | 初回ONNX検証taskは `task087` にする。 | 全公開例が3x3固定の180度回転で、static shapeの `Gather` 2段で表現できる。 | 最小の手書きONNX baselineとして以後の検証基準にした。 |
| 2026-06-05 | exp002ではstrict static filterを維持し、6500未達でもKaggle submitしない。 | 高スコア公開artifactは `Compress`, functions, dynamic shapeで大量rejectされ、許容範囲を再確認しないまま提出するとrule違反/不安定化リスクがある。 | local estimate `6282.08` の暫定bundleを保存。次PDCAはofficial rules再監査またはtop expensive task rewrite。 |
