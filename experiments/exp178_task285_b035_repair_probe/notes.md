# exp178_task285_b035_repair_probe

## 目的

exp174でpublic-zeroと推定されたtask285を、exp_b035のdistinct full-local-valid rawで修復する。

## 結果

- status: `submitted_complete`
- expected_gain_if_public_pass: `12.112175`
- expected_lb_if_public_pass: `6005.932175`
- zip sha256: `8b720af85314391f3b2925eb4fff4020d1b500aa4f84a69acb57e937e080ea6d`
- kaggle_ref: `53524248`
- public_lb: `6005.93`
- observed_gain_vs_base: `12.110000000000582`
- interpretation: 期待値と一致してrepair成功。current public LB bestを更新。

## Repair Validation

- task285: `265_pass_0_fail`, cost `395468`, sha `a7b6460a76896e8eabafd5d99bbac88a6a49d21bcf62107026bdf33a4e0f0c22`

## 判断

採用。current public LB bestをexp178へ更新する。高risk public/source blend repairなのでprivate robustness riskは継続して明記する。

## リスク

high: exp_b035 public/source blend candidate; public-zero repair may not imply private robustness.
high: selected from public-zero bisection evidence and existing public sources.
