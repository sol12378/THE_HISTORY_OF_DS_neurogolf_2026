# exp135_exp127_submission_confirmation

## Hypothesis

`docs/experiment_plan_2026-06-10.md` の Phase 0 では、未提出だった exp127 working bundle を提出すれば、local `+0.153` が LB へほぼ転写し、-352 前後の較正 gap が恒常的に残るはずである。

## Result

- Kaggle submissions を確認したところ、exp127 はすでに `2026-06-08` に提出済みだった。
- ref: `53480507`
- description: `exp127 focused surgery seventh pass micro calibration`
- public LB: `5930.55`
- source local: `6282.965236`
- exp_b025 LB `5930.40` からの増分: `+0.15`
- exp_b025 local `6282.812218` からの増分: 約 `+0.153`
- local/LB gap: 約 `-352.42`

## Interpretation

exp127 の micro delta は期待通り LB へほぼ 1:1 転写した。一方で、-352 程度の gap は維持されたため、計画書の中核仮説である「固定的な private functional failure task 群が存在する」という見方はさらに強まった。

## Decision

重複提出はしない。Phase 0 は完了扱いとし、次は Phase A の private failure 特定、または Phase C-1 の dtype/cost probe に進む。

## Leakage / Overfitting Risk

- leakage risk: low。exp127 は full-arc gated graph surgery の micro-delta。
- overfitting risk: low-to-medium。同一 task 群への繰り返し surgery だが、今回の LB 転写で micro-delta 自体は private でも概ね通っていると判断できる。
