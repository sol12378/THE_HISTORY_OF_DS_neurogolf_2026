# exp299 farm submitted best 6008.96

## 目的

exp297 が public LB `6008.96` で current best になったため、farm の提出判定基準を古い `6008.90` から `6008.96` に更新し、fresh candidate の誤提出を防ぐ。

## 変更

- `experiments/neurogolf_farm/farm_runner.py`
- `FarmConfig.submitted_best_estimate: 6008.90 -> 6008.96`

## 検証

`FarmRunner.run_smoke()` を `experiments/exp299_farm_submitted_best_6008_96/` に出力。

- submitted_best_estimate: `6008.96`
- candidate_estimate: `6009.65314718056`
- should_submit: `true`
- scope: `smoke_dummy_not_kaggle_candidate`

## 判断

tooling 基準更新のみ。smoke dummy は Kaggle candidate ではないため提出なし。
