# exp273_farm_bundle_smoke_best_by_task_visibility

## 仮説

farm smoke/result に best-by-task 件数と重複task件数を出せば、提出前 bundle の1 task 1 model制約違反を早く検知できる。

## 実施

- 変更対象は `experiments/neurogolf_farm/farm_runner.py` のみ。
- `bundle_smoke` に `best_by_task_count` と `duplicate_task_count` を追加。

## 結果

- `accepted_count=1`
- `best_by_task_count=1`
- `duplicate_task_count=0`
- farm smoke成功。

## 判断

提出なし。tooling可視化のみで、候補bundleのlocal estimate更新ではない。

## 次アクション

実bundle構築入口で `best_candidates_by_task()` を使うよう接続し、1 task 1 modelを強制する。
