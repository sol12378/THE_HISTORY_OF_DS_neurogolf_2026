# exp287_bundle_computed_local_delta

## Plan

Phase C の実候補ledger化に向けて、提出判定に使う local estimate を手入力 `local_delta` ではなく、cost から一貫計算する。

見込み:

- score delta は `ln(base_cost / candidate_cost)`
- `1000 -> 500` なら `+0.693147`
- validation fail は accepted から除外される

## Do

`experiments/neurogolf_farm/bundle_manager.py` のみ変更。

- `BundleCandidate.computed_local_delta` を追加
- `accepted_candidates()` の ranking を `computed_local_delta` 基準へ変更
- `total_local_delta()` / `best_total_local_delta()` を cost-derived delta 合計へ変更
- `submission_decision()` は既存の `best_total_local_delta()` 経由で cost-derived estimate を使う

## Check

synthetic ledger probe:

```json
{
  "expected_delta": 0.6931471805599453,
  "computed_delta": 0.6931471805599453,
  "best_total_local_delta": 0.6931471805599453,
  "candidate_estimate": 6009.59314718056,
  "failed_validation_accepted": false
}
```

farm smoke:

```json
{
  "total_local_delta": 0.6931471805599453,
  "best_total_local_delta": 0.6931471805599453,
  "candidate_estimate": 6009.59314718056,
  "submission_decision_scope": "smoke_dummy_not_kaggle_candidate"
}
```

## Act

No submit。今回の `should_submit=true` は synthetic smoke/probe 由来であり、実候補bundleではない。実提出は real candidate が full-arc gate を通り、submitted best estimate を上回った時のみ。

## Risk

- Leakage risk: なし
- Overfitting risk: なし
- Operational risk: 古い `local_delta` 列は CSV 互換のため残るが、提出判定では使わない。
