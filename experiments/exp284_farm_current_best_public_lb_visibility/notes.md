# exp284_farm_current_best_public_lb_visibility

## 目的

farm の `public_code.floor_status` が、`EXP_SUMMARY.md` に記録済みの現行 Best Public LB (`exp265`, `6008.90`) を表示できるようにする。

## 仮説

`FarmConfig.submitted_best_estimate=6008.90` は submission decision で使われているが、`PublicCodeRegistry.floor_status()` は public-code source の `result.json` だけを見ており、現行 best LB が public-code registry 側に出ない。これにより dashboard/notes 上の `max_observed_kaggle_lb` が stale または null になり、PDCA の現状把握を誤らせる。

## 実装

- `experiments/neurogolf_farm/public_code.py`
  - `PublicCodeRegistry` に `current_best_public_lb` を追加。
  - `from_experiment_results()` で `EXP_SUMMARY.md` の `| Best Public LB |` 行から現行 best public LB を読む。
  - `floor_status()` に `current_best_public_lb` を出し、`max_observed_kaggle_lb` にも含める。
  - 途中で切れていた `_float_or_none()` を修正し、数値変換に失敗した場合は `None` を返すようにした。

## 結果

`FarmRunner.run_smoke()` の `floor_status`:

```json
{
  "target_lb_floor": 6285.0,
  "submit_floor_ready": false,
  "ready_source_count": 0,
  "current_best_public_lb": 6008.9,
  "max_observed_kaggle_lb": 6008.9
}
```

`bundle_smoke.submission_decision.should_submit=true` は synthetic ledger のダミー行に対する判定であり、`submission_decision_scope=smoke_dummy_not_kaggle_candidate` のため Kaggle submit 対象ではない。

## 判定

成功。public-code floor の可視化は現行 best LB `6008.90` と整合した。

## Leakage / Overfitting risk

- Leakage risk: なし。競技データ・予測生成には触れていない。
- Overfitting risk: なし。farm tooling の表示・判断補助のみ。

## 次アクション

現行 LB 6008.90 と floor target 6285 の差は大きい。次は score-producing な候補生成に戻り、local estimate が submitted best を上回る bundle のみ submit する。
