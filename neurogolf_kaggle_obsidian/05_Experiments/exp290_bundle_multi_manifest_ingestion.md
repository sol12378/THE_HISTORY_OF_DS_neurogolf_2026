# exp290_bundle_multi_manifest_ingestion

## Plan

複数 `selected_manifest.csv` を1つの `BundleLedger` に取り込み、real candidate bundle review の入口を作る。

## Do

`bundle_manager.py` に `BundleLedger.from_selected_manifests()` を追加。

## Check

- inputs: exp260 selected manifest + exp264 selected manifest
- candidates: `35`
- accepted: `35`
- best_by_task: `35`
- duplicate_task_count: `0`
- `best_total_local_delta`: `1.4377463495672809`

## Act

提出なし。履歴manifestであり、現行best lineage に含まれる。
