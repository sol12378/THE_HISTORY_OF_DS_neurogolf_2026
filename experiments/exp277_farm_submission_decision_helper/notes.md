# exp277_farm_submission_decision_helper

## 目的

提出ポリシーでは、local estimate が提出済みベスト推定値を上回る時だけ Kaggle submit する。これを各実験の手作業判断にせず、farm ledger が `best_total_local_delta` ベースで機械的に返せるようにする。

## 変更

- `experiments/neurogolf_farm/bundle_manager.py` に `BundleLedger.submission_decision(submitted_best_estimate)` を追加した。
- 返却値:
  - `submitted_best_estimate`
  - `best_total_local_delta`
  - `candidate_estimate`
  - `should_submit`

## 確認

- `py_compile` 成功。
- submitted best `6008.90` に対して:
  - deltaあり: `best_total_local_delta=0.2`, `candidate_estimate=6009.10`, `should_submit=true`
  - 精度gate落ち: `best_total_local_delta=0`, `candidate_estimate=6008.90`, `should_submit=false`
- `FarmRunner.run_smoke()` 成功。

## 判断

提出なし。これは farm submission decision helper の tooling 較正であり、exp265 public LB `6008.90` を上回る実候補 bundle の local estimate 更新ではない。

## リスクと次アクション

- leakage risk: なし。
- overfitting risk: なし。
- 次は `farm_runner.py` の smoke/result に `submission_decision` を標準表示し、提出可否判断をresultに残す。
