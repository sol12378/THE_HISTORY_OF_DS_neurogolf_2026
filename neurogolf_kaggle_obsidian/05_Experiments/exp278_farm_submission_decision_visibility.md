# exp278_farm_submission_decision_visibility

## 仮説

farm result に `submission_decision` を出せば、各実験の提出可否判断を共通形式で残せる。これは「local estimate更新時のみ提出」の運用ミスを減らす。

## 実施

- 変更対象は `experiments/neurogolf_farm/farm_runner.py` のみ。
- `FarmConfig.submitted_best_estimate` を追加し、既定値を exp265 public LB `6008.90` にした。
- `bundle_smoke.submission_decision` を追加。

## 結果

- smoke dummy ledger:
  - `submitted_best_estimate=6008.90`
  - `best_total_local_delta=0.1`
  - `candidate_estimate=6009.0`
  - `should_submit=true`

## 判断

提出なし。これは smoke dummy ledger の表示確認であり、実候補bundleのlocal estimate更新ではない。

## 次アクション

smoke dummy と実候補 result の混同を避けるため、smoke-only decision reason を明示する。
