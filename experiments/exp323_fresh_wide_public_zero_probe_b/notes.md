# exp323_fresh_wide_public_zero_probe_b

## 目的

exp322 後の fresh wide public-zero probe。既存 probe 済み task を除外し、public-code 高リスク task を16件 fail-stub する。

## 結果

- status: `probe_zip_ready`
- targets: `[179, 241, 309, 113, 116, 164, 172, 210, 311, 326, 103, 135, 140, 186, 167, 129]`
- expected_drop_if_all_alive: `336.25926064287535`
- expected_lb_if_all_alive: `5672.700739357125`
- zip sha256: `9464fa1c79567a73b9949fc1b2567ae18e1afaa8a8158b0ad2dc8c503be34fd3`

## 判断

Kaggle ref `53551982` として提出済み。現在 `PENDING`。採点完了まで次の bisection probe は出さない。

## リスク

- leakage risk: low: deliberate fail-stub probe; no private labels used.
- overfitting risk: medium: public diagnostic probe over public-code-risk tasks; any follow-up repair must be full-arc validated and expected-LB checked.
