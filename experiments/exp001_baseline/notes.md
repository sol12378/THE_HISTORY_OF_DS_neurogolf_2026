# exp001_baseline notes

## 目的

初回提出計画に従い、スコア最大化ではなく、提出パイプラインを一周させるためのローカル環境とデータ取得可否を確認する。

## 仮説

最も単純なARCタスクを1つ選び、手書きONNXグラフでtrain pairsを完全一致させ、`onnxruntime` CPU検証と`onnx-tool`コスト計測を通せば、NeuroGolfの評価構造と提出フォーマットを安全に理解できる。

## 実施内容

- `EXP_SUMMARY.md` と `neurogolf_kaggle_obsidian/00_Index/Home.md` を確認。
- `docs/neurogolf_first_submission_plan.md` を確認。
- `nvidia-smi` でGPUドライバとVRAMを確認。
- `.venv` を作成し、`kaggle`, `onnx`, `onnxruntime`, `onnx-tool`, `numpy` を導入。
- `onnx`, `onnxruntime`, `onnx-tool`, `numpy` のimportを確認。
- Kaggle APIでコンペファイル一覧取得を試行。

## 公式仕様確認

- 評価: 正解したtaskごとに `max(1, 25 - ln(cost))`。
- cost: `params + memory bytes`。2026-05-04 updateでMACsはobjectiveから除外。
- 入力: 各gridは `[1, 10, 30, 30]` のfloat one-hot tensorに変換される。grid外はzero-hot。
- データ: `task001.json` から `task400.json`、各taskは `train`, `test`, `arc-gen` を持つ。
- 提出: `submission.zip` に最大1 task 1 ONNX。例: `task087.onnx`。
- 制約: static shape必須。`Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`, `Compress`は禁止。各ONNXは1.44MB以下。

## 結果

- GPU: OS/driverからは利用可能。RTX 2080系、VRAM 8GB、CUDA driver 13.1を確認。
- ONNX検証環境: `onnxruntime` は `CPUExecutionProvider` が利用可能。
- Kaggle API: 認証通過。
- データ: `data/raw` は触らず、`data/external/neurogolf-2026` に取得・展開。
- 最小task: `task087` を選択。公開例は固定3x3 gridの180度回転。
- ONNX: `Gather` 2段で先頭3行・3列を逆順化する手書きグラフを作成。
- ローカル検証: ARC-AGI `5 pass / 0 fail`、ARC-GEN `261 pass / 0 fail`。
- 推定cost: memory `36000` bytes + params `60`。
- 推定点: `14.507060503242691`。
- 提出物: `experiments/exp001_baseline/submission.zip` を作成。Kaggle submitは未実施。

## GPU判断

初回計画ではGPUはほぼ使わず、CPUでONNXの正しさ検証とコスト計測を行う想定。GPUを使う段階は、次フェーズ以降の勾配学習が必要な少数タスクに限るのが妥当。

現時点ではGPUドライバは問題なく見えているが、プロジェクトの`.venv`にはPyTorch CUDA環境をまだ構築していない。GPU学習を行う場合は、CUDA対応PyTorchまたは対象フレームワークを別途導入してから `torch.cuda.is_available()` で確認する。

## Leakage Risk

public train/test/arc-genに対する既知taskの手書きONNXであり、リークリスクは低い。ただしprivate validationにも同一規則が当てはまる保証はない。

## Overfitting Risk

task087に固定した3x3 rot180であり、汎化性能を競う実験ではない。Public LBだけを見て多数taskへ場当たり的に広げると過適合的な運用になるため、公式utilityでのローカル検証と仕様理解を優先する。

## 次アクション

- この単一taskの `submission.zip` を提出するか判断する。
- task087のcostをさらに下げられるか検討する。
- 同じ手順で `task140`, `task150`, `task155`, `task380` などの単純変換taskを追加する。
