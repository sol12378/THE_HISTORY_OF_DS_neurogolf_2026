# exp321_task243_101_franksunp_repair

## 目的

task101/task243 の public-zero repair として、exp320 で top だった `franksunp_blended_best` 候補を exp297 に差し替える。

## 結果

- status: `completed`
- targets: `[101, 243]`
- expected_public_gain_if_all_fixed: `27.779018459001815`
- expected_public_lb_if_all_fixed: `6036.739018459002`

## Candidate Gate
- task101: validation `266_pass_0_fail`, status `accepted_for_publiczero_repair`, cost `65875`, accepted `True`
- task243: validation `265_pass_0_fail`, status `accepted_for_publiczero_repair`, cost `67878`, accepted `True`

## 判断

Kaggle ref `53550725` は COMPLETE、Public LB は `6008.96`。期待 `6036.739018459002` には届かず、base exp297 と同点だった。

この `franksunp_blended_best` repair bundle は採用しない。task101/task243 は fail-stub でも repair でも public score に影響せず、この source では recoverable public-zero gain にならない。非 public-scoring、public 側同値、または local full-arc では見えない mismatch の可能性を残す。

## リスク

- leakage risk: medium-high: repair candidates come from public-code/raw blend artifacts. They are used only after full-arc validation, but private robustness remains uncertain.
- overfitting risk: medium: target selection used public diagnostic probes; acceptance is gated by full-arc validation and expected-LB accounting, not by adopting arbitrary public-score noise.
