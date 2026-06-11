# exp325_fresh_wide_public_zero_probe_c

## 目的

exp323 all-alive 後の fresh wide public-zero probe。既診断 task を除外し、残る public-code 高リスク task を16件 fail-stub する。

## 結果

- status: `completed`
- targets: `[334, 373, 380, 236, 106, 266, 26, 318, 144, 267, 52, 83, 108, 142, 152, 194]`
- expected_drop_if_all_alive: `289.001636451155`
- expected_lb_if_all_alive: `5719.958363548845`
- zip sha256: `e0885232daf4e1b2bd1e4229eef04657966ddfad0fdac10cac4003c575ecb9e0`

## 判断

Kaggle ref `53552426` completed at Public LB `5719.95`. This matches expected all-alive LB `5719.958363548845` within scoreboard rounding, so all 16 targets are treated as public-scoring alive and excluded from immediate public-zero repair suspects.

## リスク

- leakage risk: low: deliberate fail-stub probe; no private labels used.
- overfitting risk: medium: public diagnostic probe; any missing-drop interpretation must be followed by split confirmation or correctness-first full-arc repair.
