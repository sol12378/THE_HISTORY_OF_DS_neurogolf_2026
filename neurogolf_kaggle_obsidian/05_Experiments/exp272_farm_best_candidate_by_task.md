# exp272_farm_best_candidate_by_task

## 仮説

accepted candidate が同一taskに複数ある場合、実bundleにはbest 1件だけを入れるべきである。これを farm の ledger helper として明示すれば、recolor_direct / recolor_cast のような同一delta候補で安い候補を安定して選べる。

## 実施

- 変更対象は `experiments/neurogolf_farm/bundle_manager.py` のみ。
- `BundleLedger.best_candidates_by_task()` を追加。

## 結果

- probe accepted count: `3`
- best-by-task count: `2`
- `task_recolor` は `recolor_direct(cost 45)` が残り、`recolor_cast(cost 141)` は除外された。
- farm smoke成功。

## 判断

提出なし。tooling dedupe 較正のみで、候補bundleのlocal estimate更新ではない。

## 次アクション

farm smoke / result に best-by-task count を標準出力し、提出前 bundle で重複taskがないかを可視化する。
