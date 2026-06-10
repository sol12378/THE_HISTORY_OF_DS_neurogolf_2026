# exp292_submission_decision_counts

## Plan

現行best lineageを除外した後に、fresh候補が何件残っているかを `submission_decision` だけで監査できるようにする。

見込み:

- 履歴manifestは `accepted_count>0` かつ `should_submit=true` になり得る。
- lineage除外後は fresh候補がなければ `accepted_count=0` / `should_submit=false` になる。

## Do

`experiments/neurogolf_farm/bundle_manager.py` のみ変更。

- `BundleLedger.submission_decision()` に `accepted_count` と `best_by_task_count` を追加。

## Check

入力:

- exp254 / exp256 / exp258 / exp260 / exp262 partial / exp264 selected manifests

除外なし:

```json
{
  "accepted_count": 104,
  "best_by_task_count": 104,
  "best_total_local_delta": 2.6284155220455716,
  "candidate_estimate": 6011.528415522045,
  "should_submit": true
}
```

現行best lineage sourceを除外後:

```json
{
  "accepted_count": 0,
  "best_by_task_count": 0,
  "best_total_local_delta": 0,
  "candidate_estimate": 6008.9,
  "should_submit": false
}
```

## Act

No submit。fresh候補は0件で、local estimateはsubmitted best `6008.90` を更新しない。

## Risk

- Leakage risk: なし
- Overfitting risk: なし
- Operational risk: lineage source の除外漏れがあると履歴deltaが見える。`accepted_count` / `best_by_task_count` を必ず確認する。
