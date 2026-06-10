# exp172_task133_158_b035_repair_probe

## 目的

exp169でpublic-zeroと推定されたtask133/task158を、exp_b035のdistinct full-local-valid rawで修復する。

## 結果

- Kaggle ref: `53523884`
- public LB: `5993.82`
- expected_lb_if_public_pass: `5993.818803`
- observed_gain_vs_exp166: `+25.64`
- task133 validation: `267_pass_0_fail`, cost `196688`
- task158 validation: `266_pass_0_fail`, cost `193266`

## 判断

採用。current public LB bestをexp172へ更新する。

## リスク

exp_b035 public/source blend由来のrepairであり、private robustness riskは高い。public bestとして扱うが、private-safeとは扱わない。
