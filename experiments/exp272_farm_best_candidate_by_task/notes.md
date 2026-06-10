# exp272_farm_best_candidate_by_task

## 目的

exp271で accepted candidate の順序を cost-aware にしたが、同一taskの複数accepted候補がそのまま返る問題が残っていた。実bundleでは1 task 1 modelなので、taskごとのbest候補だけを選べる helper を追加する。

## 変更

- `experiments/neurogolf_farm/bundle_manager.py` に `BundleLedger.best_candidates_by_task()` を追加した。
- 既存の `accepted_candidates()` の順序、つまり `local_delta` 降順・`candidate_cost` 昇順を使い、taskごとに最初の1件だけを保持する。

## 確認

- `py_compile` 成功。
- probeでは accepted 3件のうち、best-by-task は2件に絞られた。
- `task_recolor` は `recolor_cast(cost 141)` ではなく `recolor_direct(cost 45)` が残った。
- `FarmRunner.run_smoke()` 成功。

## 判断

提出なし。これは bundle dedupe の tooling 較正であり、exp265 public LB `6008.90` を上回る候補 bundle の local estimate 更新ではない。

## リスクと次アクション

- leakage risk: なし。
- overfitting risk: なし。
- 次は farm smoke の `bundle_smoke` でも `best_candidates_by_task()` を報告し、運用時に重複task数を見えるようにする。
