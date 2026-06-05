# Decision Log

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-06-05 | exp001はCPU ONNX検証を優先し、GPU学習は次フェーズ以降に回す。 | 初回提出計画はパイプライン一周が目的で、`onnxruntime` CPU検証と`onnx-tool`計測で十分。 | 依存は軽量ONNX系から構築。GPUはdriver確認のみ行い、Kaggle認証後にデータ取得へ進む。 |
| 2026-06-05 | 初回ONNX検証taskは `task087` にする。 | 全公開例が3x3固定の180度回転で、static shapeの `Gather` 2段で表現できる。 | `task087.onnx` と単一task `submission.zip` を作成。Kaggle submitは判断待ち。 |
