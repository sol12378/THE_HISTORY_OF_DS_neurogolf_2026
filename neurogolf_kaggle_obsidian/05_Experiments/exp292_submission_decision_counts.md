# exp292_submission_decision_counts

## Plan

`submission_decision` に候補件数を出し、fresh候補が残っているかを判定しやすくする。

## Do

`bundle_manager.py` の `submission_decision()` に `accepted_count` と `best_by_task_count` を追加。

## Check

- 最近のbypass manifest全体: `accepted_count=104`, `best_total_local_delta=2.6284155220455716`, `should_submit=true`
- 現行best lineage source除外後: `accepted_count=0`, `best_total_local_delta=0`, `should_submit=false`

## Act

提出なし。fresh local estimate改善なし。
