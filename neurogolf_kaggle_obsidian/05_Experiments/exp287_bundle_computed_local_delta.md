# exp287_bundle_computed_local_delta

## Plan

実候補ledgerの提出判断を、手入力 `local_delta` ではなく cost 由来の `ln(base_cost / candidate_cost)` に統一する。

## Do

`bundle_manager.py` のみ変更。

- `computed_local_delta` を追加
- accepted ranking / total / best_total / submission decision を cost-derived delta に接続

## Check

- `1000 -> 500`: `computed_delta=0.6931471805599453`
- validation fail candidate: accepted されない
- farm smoke: `candidate_estimate=6009.59314718056`
- scope: `smoke_dummy_not_kaggle_candidate`

## Act

提出なし。synthetic ledger のみ。real candidate ではない。
