# exp289_bundle_selected_manifest_ingestion

## Plan

既存 `selected_manifest.csv` を `BundleLedger` に直接取り込み、real candidate manifest を cost-derived submit decision に接続する。

## Do

`bundle_manager.py` に `BundleLedger.from_selected_manifest()` を追加。

## Check

- input: `experiments/exp264_current_rank321_400_fullarc_bypass_sweep/selected_manifest.csv`
- candidates: `8`
- accepted: `8`
- `best_total_local_delta`: `0.5976596441240898`
- `candidate_estimate`: `6009.497659644124`

## Act

提出なし。exp264/exp265 delta は既に現行bestに含まれるため、今回は ingestion 検証のみ。
