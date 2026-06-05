# Decision Log

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-06-05 | exp001はCPU ONNX検証を優先し、GPU学習は次フェーズ以降に回す。 | 初回提出計画はパイプライン一周が目的で、ONNX Runtime CPU検証と公式cost確認で十分。 | `task087.onnx` と単一task `submission.zip` を作成し、Kaggle submit ref `53383536` / public LB `14.50` を記録。 |
| 2026-06-05 | 初回ONNX検証taskは `task087` にする。 | 公開例が3x3固定の180度回転で、static shapeの `Gather` 2段で表現できる。 | 最小の手書きONNX baselineとして以後の検証基準にした。 |
| 2026-06-05 | exp002ではstrict static filterを維持し、6500未達でもKaggle submitしない。 | 高スコア公開artifactは `Compress`, functions, dynamic shapeでrejectされるものが多く、許容範囲を確認しないまま提出するとrule違反/不安定化リスクがある。 | local estimate `6282.08` の暫定bundleを保存。次PDCAはofficial rules再確認またはtop expensive task rewrite。 |
| 2026-06-05 | CUDAを先行導入し、GPUはrouting/rankingに使う。 | PyTorch CUDA smoke testが通り、RTX 2080 SUPERで軽量CNN学習が実行できた。提出ONNXの採用判定はCPU official-like validationに固定する。 | `exp010_cuda_gpu_setup` と `exp011_gpu_route_classifier` を追加。GPU学習は有効化済み。 |
| 2026-06-05 | exp012のlookup改善は提出候補にしない。 | `signature_scatter_lookup` はlocal estimateを `6296.30` まで上げたが、arc-gen labelを利用するためleakage/overfitting riskが高く、6500にも届かない。 | exp012はlocal上限探索として保存。次はgraph surgeryと小型CNN/conv templateのstrict採用へ進む。 |
