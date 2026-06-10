# exp311_submission_decision_primitive_counts

## Plan

Phase C の `grid_sample` 実候補が提出判断に入った時、primitive別 accepted 件数を decision で見られるようにする。見込みコスト削減は直接なし。

## Do

`bundle_manager.py` の `submission_decision()` に `accepted_count_by_primitive` を追加。

## Check

- accepted_count: `1`
- accepted_count_by_primitive: `{"grid_sample": 1}`
- should_submit: `true`

## Act

synthetic ledger の metadata 更新のみで実候補bundleのlocal estimate改善ではないため提出なし。次は `grid_sample` supplier を実task候補へ接続する。
