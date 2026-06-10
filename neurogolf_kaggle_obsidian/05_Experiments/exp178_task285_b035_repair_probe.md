# exp178_task285_b035_repair_probe

## 目的

exp174でpublic-zeroと推定されたtask285を、exp_b035のdistinct full-local-valid rawで修復する。

## 結果

- Kaggle ref: `53524248`
- public LB: `6005.93`
- expected_lb_if_public_pass: `6005.932175`
- observed_gain_vs_exp172: `+12.11`
- validation: `265_pass_0_fail`
- cost: `395468`

## 判断

採用。current public LB bestをexp178へ更新する。6000 public LBを突破。

## リスク

exp_b035 public/source blend由来のrepairであり、private robustness riskは高い。public bestとして扱うが、private-safeとは扱わない。
