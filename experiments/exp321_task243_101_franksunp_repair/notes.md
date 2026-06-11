# exp321_task243_101_franksunp_repair

## 目的

task101/task243 の public-zero repair として、exp320 で top だった `franksunp_blended_best` 候補を exp297 に差し替える。

## 結果

- status: `zip_ready`
- targets: `[101, 243]`
- expected_public_gain_if_all_fixed: `27.779018459001815`
- expected_public_lb_if_all_fixed: `6036.739018459002`

## Candidate Gate
- task101: validation `266_pass_0_fail`, status `accepted_for_publiczero_repair`, cost `65875`, accepted `True`
- task243: validation `265_pass_0_fail`, status `accepted_for_publiczero_repair`, cost `67878`, accepted `True`

## 判断

Kaggle ref `53550725` として提出済み。現在 `PENDING`。

## リスク

- leakage risk: medium-high: repair candidates come from public-code/raw blend artifacts. They are used only after full-arc validation, but private robustness remains uncertain.
- overfitting risk: medium: target selection used public diagnostic probes; acceptance is gated by full-arc validation and expected-LB accounting, not by adopting arbitrary public-score noise.
