# exp273_farm_bundle_smoke_best_by_task_visibility

## 目的

exp272で `BundleLedger.best_candidates_by_task()` を追加したが、farm smoke/result には accepted 全件数しか出ていなかった。提出前 bundle の task重複を早く検知できるよう、smoke result に best-by-task 件数と重複件数を出す。

## 変更

- `experiments/neurogolf_farm/farm_runner.py` の `bundle_smoke` に以下を追加した。
  - `best_by_task_count`
  - `duplicate_task_count`

## 確認

- `py_compile` 成功。
- `FarmRunner.run_smoke()` 成功。
- exp273 smokeでは `accepted_count=1`, `best_by_task_count=1`, `duplicate_task_count=0`。

## 判断

提出なし。これは farm smoke 可視化の tooling 較正であり、exp265 public LB `6008.90` を上回る候補 bundle の local estimate 更新ではない。

## リスクと次アクション

- leakage risk: なし。
- overfitting risk: なし。
- 次は実候補生成側が `best_candidates_by_task()` を使っているかを確認し、未接続なら bundle export の入口で1 task 1 modelを強制する。
