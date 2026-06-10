# exp311 submission decision primitive counts

## 目的

Phase C の `grid_sample` 実候補が提出判断に入った時、primitive別 accepted 件数を decision で見られるようにする。

## 変更

- `experiments/neurogolf_farm/bundle_manager.py`
- `submission_decision()` に `accepted_count_by_primitive` を追加。

## 結果

synthetic ledger:

- accepted_count: `1`
- best_by_task_count: `1`
- accepted_count_by_primitive: `{"grid_sample": 1}`
- should_submit: `true`

## 判断

synthetic ledger の metadata 更新のみで実候補bundleのlocal estimate改善ではないため提出なし。次は `grid_sample` supplier を実task候補へ接続する。

## リスク

- leakage risk: low。提出候補なし。
- overfitting risk: low。synthetic ledger の metadata 追加のみ。
