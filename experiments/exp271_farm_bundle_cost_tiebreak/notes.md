# exp271_farm_bundle_cost_tiebreak

## 目的

exp270で `recolor_direct` と `recolor_cast` の cost 差を IR local estimate に表現できるようになった。次に、その差が bundle candidate の採用順に反映されるかを確認し、同一 `local_delta` なら低cost候補を優先する。

## 変更

- `experiments/neurogolf_farm/bundle_manager.py` の `BundleLedger.accepted_candidates()` を、accepted候補を返すだけでなく以下の順に sort するよう変更した。
  - `local_delta` 降順
  - `candidate_cost` 昇順
  - `task_id` 昇順

## 確認

- `py_compile` 成功。
- 入力順を `recolor_cast` -> `recolor_direct` にしても、返却順は `recolor_direct(cost 45)` -> `recolor_cast(cost 141)` になった。
- `FarmRunner.run_smoke()` 成功。

## 判断

提出なし。これは bundle ranking の tooling 較正であり、exp265 public LB `6008.90` を上回る候補 bundle の local estimate 更新ではない。

## リスクと次アクション

- leakage risk: なし。
- overfitting risk: なし。
- 次は同一taskに複数accepted候補がある場合、taskごとにbest candidateだけを選ぶ helper を追加するか確認する。現状は accepted 全件を返すため、bundle構築側で重複taskを扱う必要がある。
