# exp310 bundle candidate primitive kind

## 目的

Phase C の `grid_sample` 実候補を ledger 上で追跡できるよう、`BundleCandidate` に `primitive_kind` を追加する。

## 変更

- `experiments/neurogolf_farm/bundle_manager.py`
- `BundleCandidate` に `primitive_kind` を追加。
- selected manifest ingestion で `primitive_kind` を取り込む。
- `write_csv()` に `primitive_kind` 列を追加。

## 結果

- synthetic csv_primitive_kind: `grid_sample`
- accepted_count: `1`
- best_total_local_delta: `4.710530701645918`

## 判断

synthetic ledger の metadata 更新のみで実候補bundleのlocal estimate改善ではないため提出なし。次は `grid_sample` supplier を実task候補へ接続する。

## リスク

- leakage risk: low。提出候補なし。
- overfitting risk: low。synthetic ledger の metadata 追加のみ。
