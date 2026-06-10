# exp290_bundle_multi_manifest_ingestion

## Plan

Phase C の real candidate bundle review に向けて、複数の `selected_manifest.csv` を1つの `BundleLedger` に取り込めるようにする。

見込み:

- 複数 sweep の候補をまとめて cost-derived estimate へ流せる。
- 同一 task の重複候補は既存 `best_candidates_by_task()` で最良候補だけを評価できる。

## Do

`experiments/neurogolf_farm/bundle_manager.py` のみ変更。

- `BundleLedger.from_selected_manifests()` を追加。
- 各 manifest は既存 `from_selected_manifest()` で読み、候補を連結する。

## Check

入力:

- `experiments/exp260_current_rank141_220_fullarc_bypass_sweep/selected_manifest.csv`
- `experiments/exp264_current_rank321_400_fullarc_bypass_sweep/selected_manifest.csv`

結果:

```json
{
  "candidate_count": 35,
  "accepted_count": 35,
  "best_by_task_count": 35,
  "duplicate_task_count": 0,
  "best_total_local_delta": 1.4377463495672809,
  "candidate_estimate": 6010.337746349567
}
```

Accuracy gate:

- 全 accepted rows は `266_pass_0_fail` に正規化済み。

## Act

No submit。入力に使った exp260/exp264 系 delta は既に現行 best lineage に含まれる履歴候補であり、新規 local estimate 更新ではない。

## Risk

- Leakage risk: なし。既存 full-arc manifest の読み込みのみ。
- Overfitting risk: なし。新規候補採用や提出はしていない。
- Operational risk: 複数履歴manifestをまとめると `should_submit=true` が出るが、履歴deltaを再提出対象と誤読しない。
