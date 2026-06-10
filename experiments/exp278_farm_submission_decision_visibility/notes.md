# exp278_farm_submission_decision_visibility

## 目的

exp277で `BundleLedger.submission_decision()` を追加したが、farm smoke/result には表示されていなかった。提出可否判断を実験artifactへ残すため、`bundle_smoke` に `submission_decision` を標準表示する。

## 変更

- `experiments/neurogolf_farm/farm_runner.py` の `FarmConfig` に `submitted_best_estimate` を追加した。
- 既定値は現current bestの exp265 public LB `6008.90`。
- `bundle_smoke` に `ledger.submission_decision(self.config.submitted_best_estimate)` を追加した。

## 確認

- `py_compile` 成功。
- `FarmRunner.run_smoke()` 成功。
- smoke dummy候補では `submitted_best_estimate=6008.90`, `best_total_local_delta=0.1`, `candidate_estimate=6009.0`, `should_submit=true`。

## 判断

Kaggle提出なし。`should_submit=true` は smoke dummy ledger に対する機械判定であり、実候補 bundle の local estimate 更新ではない。

## リスクと次アクション

- leakage risk: なし。
- overfitting risk: なし。
- 次は smoke dummy と実候補 result を混同しないよう、`status` または `submission_reason` に smoke-only であることを明示する。
