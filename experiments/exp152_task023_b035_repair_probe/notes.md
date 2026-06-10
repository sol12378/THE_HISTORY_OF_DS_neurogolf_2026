# exp152_task023_b035_repair_probe

## Hypothesis

exp150でpublic-zeroと推定したtask023を、full validation OKのexp_b035 rawへ差し替えるとpublic LBが回復する。

## Result

- status: `public_lb_improved`
- base_exp: `experiments\exp144_task018_b035_repair_probe`
- candidate_source: `exp_b035_new_source_full_arc_blend`
- candidate_validation: `266_pass_0_fail`
- candidate_cost: `72612`
- expected_public_lb_if_pass: `5956.287114999999`
- Kaggle ref: `53521952`
- public LB: `5956.28`
- delta_vs_base_public_lb: `+13.80`
- interpretation: task023 exp_b035 repairがpublicで通り、current public LB bestを更新。
- zip sha256: `6e36d2b7eaddeeea91466c9042c2bbb9f9e77e74cfa7498a0f271d9030ae25a4`

## Decision

adopt_as_current_public_lb_best_with_high_risk_flag

## Leakage / Overfitting Risk

high: candidate is from an existing public/source blend and may be public-overfit.
high: task023 was identified through public bisection; repair must be LB-probed and not assumed private-safe.
