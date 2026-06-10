# exp271_farm_bundle_cost_tiebreak

## 仮説

`recolor_direct` と `recolor_cast` の cost 差を計算できても、accepted candidate の順序が元順のままだと、同一 `local_delta` の候補で低cost側を優先できない。bundle ranking は cost-aware であるべき。

## 実施

- 変更対象は `experiments/neurogolf_farm/bundle_manager.py` のみ。
- `accepted_candidates()` を `local_delta desc`, `candidate_cost asc`, `task_id asc` で sort。

## 結果

- probeでは、入力順が `recolor_cast(cost 141)` -> `recolor_direct(cost 45)` でも、返却順は `recolor_direct` -> `recolor_cast`。
- farm smoke成功。

## 判断

提出なし。tooling ranking 較正のみで、候補bundleのlocal estimate更新ではない。

## 次アクション

同一taskの複数accepted候補について、taskごとにbest 1件だけを返す helper を追加するか検討する。これは実bundle構築時の重複防止に効く。
