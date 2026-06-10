# exp143_task018_beicicc_repair_probe

## Hypothesis

task018はexp127でpublic-zeroと推定される。full local validationを通るbeicicc候補へ単独差し替えすれば、public LBを回収できる可能性がある。

## Result

- status: `no_public_gain`
- candidate validation: `266_pass_0_fail`
- candidate cost from manifest: `33897`
- expected_lb_if_public_pass: `5945.1189182061225`
- zip sha256: `1337db064a098bbea27e5386e86f1dee3eef23102a4042c4aa9ff08b2ba1aff2`

## Decision

Kaggle ref `53520866` として提出し、public LB は `5930.55`。exp127から改善なし。

beicicc task018候補は full local validation `266_pass_0_fail` だが、publicでは0点のまま。採用しない。

## Leakage / Overfitting Risk

high: beicicc public-code source collapsed in exp130; this is a single-task calibration probe, not a final private-safe repair yet.
high: public pass would not prove private robustness.
