# exp291_bundle_exclude_submitted_sources

## Plan

履歴manifestが `should_submit=true` を出す問題を防ぐため、現行best lineage の `source_exp` を除外してから提出判定する。

## Do

`bundle_manager.py` の `BundleLedger.from_selected_manifests()` に `exclude_sources` を追加。

## Check

- 除外なし: 35 accepted, `best_total_local_delta=1.4377463495672809`, `should_submit=true`
- exp260/exp264 source 除外後: 0 accepted, `best_total_local_delta=0`, `should_submit=false`

## Act

提出なし。除外後に fresh local estimate 改善なし。
