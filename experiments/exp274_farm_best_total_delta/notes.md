# exp274_farm_best_total_delta

## 目的

exp272で taskごとのbest候補を選べるようにしたが、`total_local_delta()` は accepted 全件を合計するため、同一taskに複数accepted候補があると local estimate を過大計上し得る。提出判断に使える best-by-task ベースの delta 合計を追加する。

## 変更

- `experiments/neurogolf_farm/bundle_manager.py` に `BundleLedger.best_total_local_delta()` を追加した。
- 既存の `total_local_delta()` は全accepted監査用として残した。

## 確認

- `py_compile` 成功。
- 重複候補probe:
  - accepted count: `3`
  - best-by-task count: `2`
  - total local delta: `1.4`
  - best total local delta: `0.9`
- `FarmRunner.run_smoke()` 成功。

## 判断

提出なし。これは bundle local-delta accounting の tooling 較正であり、exp265 public LB `6008.90` を上回る候補 bundle の local estimate 更新ではない。

## リスクと次アクション

- leakage risk: なし。
- overfitting risk: なし。
- 次は `farm_runner.py` の `bundle_smoke` に `best_total_local_delta` を表示し、提出判断用の値を標準可視化する。
