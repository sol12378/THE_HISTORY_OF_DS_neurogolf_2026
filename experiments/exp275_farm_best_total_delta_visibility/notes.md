# exp275_farm_best_total_delta_visibility

## 目的

exp274で `BundleLedger.best_total_local_delta()` を追加したが、farm smoke/result には表示されていなかった。提出判断に使う dedupe済み local delta を標準で見えるようにする。

## 変更

- `experiments/neurogolf_farm/farm_runner.py` の `bundle_smoke` に `best_total_local_delta` を追加した。

## 確認

- `py_compile` 成功。
- `FarmRunner.run_smoke()` 成功。
- exp275 smokeでは `accepted_count=1`, `best_by_task_count=1`, `duplicate_task_count=0`, `total_local_delta=0.1`, `best_total_local_delta=0.1`。

## 判断

提出なし。これは farm smoke 可視化の tooling 較正であり、exp265 public LB `6008.90` を上回る候補 bundle の local estimate 更新ではない。

## リスクと次アクション

- leakage risk: なし。
- overfitting risk: なし。
- 次は `write_notes()` 側の説明も現在の output-bytes / best-by-task policy に合わせ、古い `Where` reject 表現や accepted count だけの説明を更新する。
