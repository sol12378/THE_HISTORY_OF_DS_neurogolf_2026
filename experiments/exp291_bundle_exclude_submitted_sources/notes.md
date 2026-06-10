# exp291_bundle_exclude_submitted_sources

## Plan

Phase C の real manifest review で、現行 best lineage に含まれる履歴 source を誤って再提出候補にしないよう、`BundleLedger.from_selected_manifests()` に source 除外を追加する。

見込み:

- 履歴manifestをまとめて読んでも、`exclude_sources` に現行 best lineage の `source_exp` を渡せば fresh候補だけが残る。
- 履歴delta由来の `should_submit=true` を防げる。

## Do

`experiments/neurogolf_farm/bundle_manager.py` のみ変更。

- `BundleLedger.from_selected_manifests(paths, exclude_sources=...)` を追加。
- `source_exp` が除外集合に入る候補は ledger に入れない。

## Check

入力:

- `experiments/exp260_current_rank141_220_fullarc_bypass_sweep/selected_manifest.csv`
- `experiments/exp264_current_rank321_400_fullarc_bypass_sweep/selected_manifest.csv`

除外なし:

```json
{
  "candidate_count": 35,
  "accepted_count": 35,
  "best_total_local_delta": 1.4377463495672809,
  "should_submit": true
}
```

現行best lineage sourceを除外後:

```json
{
  "candidate_count": 0,
  "accepted_count": 0,
  "best_total_local_delta": 0,
  "candidate_estimate": 6008.9,
  "should_submit": false
}
```

## Act

No submit。除外後に fresh local estimate 改善はない。

## Risk

- Leakage risk: なし
- Overfitting risk: なし
- Operational risk: `exclude_sources` の指定漏れがあると履歴deltaが再び見える。現行best lineage は明示的に渡す。
