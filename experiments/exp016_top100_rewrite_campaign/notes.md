# exp016_top100_rewrite_campaign notes

- local estimate: `6479.3830863527055`
- status: `below_target`
- Kaggle submitは行わない。
- top400 campaignで `195` taskを改善したが、6500まで `20.616913647294496` 不足。
- 採用改善は `signature_scatternd_lookup` が中心で、leakage/overfitting riskは高い。
- 次PDCAではremaining high cost taskの個別graph surgeryを優先する。
