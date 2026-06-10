# exp275_farm_best_total_delta_visibility

## 仮説

提出判断では同一task重複を除いた `best_total_local_delta` を見るべきである。farm smoke/result にこれを表示すれば、local estimate 更新判定の根拠が明確になる。

## 実施

- 変更対象は `experiments/neurogolf_farm/farm_runner.py` のみ。
- `bundle_smoke` に `best_total_local_delta` を追加。

## 結果

- `accepted_count=1`
- `best_by_task_count=1`
- `duplicate_task_count=0`
- `total_local_delta=0.1`
- `best_total_local_delta=0.1`
- farm smoke成功。

## 判断

提出なし。tooling可視化のみで、候補bundleのlocal estimate更新ではない。

## 次アクション

`FarmRunner.write_notes()` の古い説明を、現在の output bytes / best-by-task / best total delta 方針へ更新する。
