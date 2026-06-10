# exp274_farm_best_total_delta

## 仮説

同一taskに複数accepted候補がある場合、accepted全件の `local_delta` を合計すると提出前 local estimate を過大評価する。best-by-task後の delta 合計を別に持つ必要がある。

## 実施

- 変更対象は `experiments/neurogolf_farm/bundle_manager.py` のみ。
- `BundleLedger.best_total_local_delta()` を追加。

## 結果

- probe accepted count: `3`
- best-by-task count: `2`
- total local delta: `1.4`
- best total local delta: `0.9`
- farm smoke成功。

## 判断

提出なし。tooling accounting 較正のみで、候補bundleのlocal estimate更新ではない。

## 次アクション

`farm_runner.py` の smoke/result で `best_total_local_delta` を標準表示し、提出判断ではこの値を参照する。
