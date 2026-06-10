# exp312_submission_decision_primitive_delta

## Plan

Phase C の `grid_sample` 実候補が提出判断に入った時、primitive別 local delta 寄与を decision で見られるようにする。見込みコスト削減は直接なし。

## Do

`bundle_manager.py` の `submission_decision()` に `best_delta_by_primitive` を追加。

## Check

- accepted_count: `1`
- best_delta_by_primitive: `{"grid_sample": 4.710530701645918}`
- grid_sample_delta: `4.710530701645918`
- should_submit: `true`

## Act

synthetic ledger の metadata 更新のみで実候補bundleのlocal estimate改善ではないため提出なし。次は `grid_sample` supplier を実task候補へ接続する。
