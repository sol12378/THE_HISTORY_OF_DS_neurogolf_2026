# exp293_bundle_fresh_submission_decision

## Plan

次の real candidate manifest 生成後に、現行best lineage除外と提出判定を1回で実行できるようにする。

見込み:

- 履歴manifest全体では `should_submit=true` になり得る。
- `fresh_submission_decision()` では `exclude_sources` を必須にして、fresh候補だけで `should_submit` を出す。

## Do

`experiments/neurogolf_farm/bundle_manager.py` のみ変更。

- `BundleLedger.fresh_submission_decision(paths, submitted_best_estimate=..., exclude_sources=...)` を追加。
- 内部で `from_selected_manifests(..., exclude_sources=...)` と `submission_decision()` を接続。
- 出力に `excluded_sources` と `input_manifest_count` を含める。

## Check

入力:

- exp254 / exp256 / exp258 / exp260 / exp262 partial / exp264 selected manifests

履歴全体:

```json
{
  "accepted_count": 104,
  "best_total_local_delta": 2.6284155220455716,
  "candidate_estimate": 6011.528415522045,
  "should_submit": true
}
```

fresh-only:

```json
{
  "accepted_count": 0,
  "best_by_task_count": 0,
  "best_total_local_delta": 0,
  "candidate_estimate": 6008.9,
  "should_submit": false,
  "input_manifest_count": 6
}
```

## Act

No submit。fresh-only accepted候補は0件で、submitted best `6008.90` を更新しない。

## Risk

- Leakage risk: なし
- Overfitting risk: なし
- Operational risk: `exclude_sources` は必須引数。現行best lineage をEXP_SUMMARY/LB trackingと照合して渡す。
