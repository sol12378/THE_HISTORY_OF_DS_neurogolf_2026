# exp288_bundle_csv_computed_delta

## Plan

BundleLedger CSV に、提出判定で使う cost-derived delta を明示する。

## Do

`bundle_manager.py` の `write_csv()` に `computed_local_delta` 列を追加。

## Check

- CSV first row `computed_local_delta`: `0.6931471805599453`
- `bundle_smoke.best_total_local_delta`: `0.6931471805599453`
- scope: `smoke_dummy_not_kaggle_candidate`

## Act

提出なし。real candidate ではなく smoke visibility の改善。
