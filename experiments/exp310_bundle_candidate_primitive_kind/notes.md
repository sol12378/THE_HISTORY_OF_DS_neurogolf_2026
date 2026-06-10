# exp310_bundle_candidate_primitive_kind

## Plan

Phase C の `grid_sample` 実候補を ledger 上で追跡できるよう、`BundleCandidate` に `primitive_kind` を追加する。見込みコスト削減は直接なし。

## Do

`bundle_manager.py` の `BundleCandidate` / manifest ingestion / CSV output に `primitive_kind` を追加。

## Check

- csv_primitive_kind: `grid_sample`
- accepted_count: `1`
- best_total_local_delta: `4.710530701645918`

## Act

synthetic ledger の metadata 更新のみで実候補bundleのlocal estimate改善ではないため提出なし。次は `grid_sample` supplier を実task候補へ接続する。
