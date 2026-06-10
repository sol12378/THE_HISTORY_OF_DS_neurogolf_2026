# exp289_bundle_selected_manifest_ingestion

## Plan

Phase C の real candidate ledger 化に向けて、既存の `selected_manifest.csv` を `BundleLedger` に直接取り込めるようにする。

見込み:

- `baseline_cost` / `candidate_cost` から `computed_local_delta=ln(base/candidate)` を計算できる。
- `validation_status=full_arc_pass` を accuracy gate 通過 (`266_pass_0_fail`) として扱える。
- `status=improved` の候補だけ accepted として提出判定へ流せる。

## Do

`experiments/neurogolf_farm/bundle_manager.py` のみ変更。

- `BundleLedger.from_selected_manifest()` を追加。
- `full_arc_pass` を `266_pass_0_fail` に正規化。
- `baseline_cost` / `candidate_cost` を cost-derived delta に接続。

## Check

入力:

- `experiments/exp264_current_rank321_400_fullarc_bypass_sweep/selected_manifest.csv`

結果:

```json
{
  "candidate_count": 8,
  "accepted_count": 8,
  "best_total_local_delta": 0.5976596441240898,
  "candidate_estimate": 6009.497659644124,
  "first_accepted_computed_local_delta": 0.2209773737690483
}
```

Accuracy gate:

- `full_arc_pass` 行を `266_pass_0_fail` に正規化して accepted。
- validation fail 行は今回の manifest には含まれていない。

## Act

No submit。入力に使った exp264/exp265 系 delta は既に現行 best `6008.90` に反映済みであり、新規 local estimate 更新ではない。今回の目的は real manifest ingestion の検証。

## Risk

- Leakage risk: なし。既存 full-arc manifest の読み込みのみ。
- Overfitting risk: なし。新規候補採用や提出はしていない。
- Operational risk: 既提出 manifest を再度 should_submit と読まない。現行 best に含まれる履歴候補は submit 対象外。
