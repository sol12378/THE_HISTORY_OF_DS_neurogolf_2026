# exp296_bundle_exclude_tasks_quarantine

## Plan

exp295 で `task048` が exp262 sweep を止めることが分かったため、farm の fresh submission review に task-level quarantine を追加する。

見込み:

- `exclude_tasks={"48"}` を指定すれば、task048由来の候補を提出判定から除外できる。
- 残り候補だけで `accepted_count` / `best_total_local_delta` / `should_submit` を計算できる。

## Do

`experiments/neurogolf_farm/bundle_manager.py` のみ変更。

- `BundleLedger.from_selected_manifests(..., exclude_tasks=...)` を追加。
- `BundleLedger.fresh_submission_decision(..., exclude_tasks=...)` に接続。
- 出力に `excluded_tasks` を含める。

## Check

synthetic manifest:

- task048: `1000 -> 500`
- task049: `1000 -> 800`

結果:

```json
{
  "no_quarantine": {
    "accepted_count": 2,
    "best_total_local_delta": 0.9162907318741551,
    "should_submit": true
  },
  "task048_quarantine": {
    "accepted_count": 1,
    "best_total_local_delta": 0.22314355131420976,
    "excluded_tasks": ["48"]
  },
  "all_tasks_quarantine": {
    "accepted_count": 0,
    "best_total_local_delta": 0,
    "should_submit": false
  }
}
```

## Act

No submit。synthetic quarantine probe のみで、実候補bundleの local estimate 更新ではない。

## Next

task048 の bad candidates を生成段階でも skip/quarantine できる経路を作り、exp262 rank221-320 の残りwindowを完走する。

## Risk

- Leakage risk: なし
- Overfitting risk: なし
- Operational risk: task quarantine は提出review側の安全弁であり、生成scriptのruntime停止そのものはまだ解決していない。
