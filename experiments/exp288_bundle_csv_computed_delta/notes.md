# exp288_bundle_csv_computed_delta

## Plan

Phase C の real candidate review に向けて、`BundleLedger.write_csv()` が提出判定と同じ cost-derived delta を出力するようにする。

見込み:

- CSV review 上の delta と `submission_decision.best_total_local_delta` が一致する。
- 古い手入力 `local_delta` が残っていても、提出判断に使う値を明示できる。

## Do

`experiments/neurogolf_farm/bundle_manager.py` のみ変更。

- CSV field に `computed_local_delta` を追加
- 各行に `candidate.computed_local_delta` を出力

## Check

farm smoke から生成した `bundle_ledger_smoke.csv`:

```json
{
  "computed_local_delta": "0.6931471805599453",
  "best_total_local_delta": 0.6931471805599453,
  "submission_decision_scope": "smoke_dummy_not_kaggle_candidate"
}
```

## Act

No submit。smoke dummy ledger の visibility 改善のみで、実候補bundleの local estimate 更新ではない。

## Risk

- Leakage risk: なし
- Overfitting risk: なし
- Operational risk: `local_delta` 列は互換目的で残る。提出判定・review は `computed_local_delta` を見る。
