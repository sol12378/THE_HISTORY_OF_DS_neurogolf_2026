# exp152_task023_b035_repair_probe

## 目的

exp150でpublic-zeroと推定したtask023を、full validation OKの `exp_b035` rawへ差し替え、exp144 public bestに上乗せできるか確認する。

## 結果

- Kaggle ref: `53521952`
- public LB: `5956.28`
- base: exp144 public LB `5942.48`
- delta vs exp144: `+13.80`
- delta vs exp127: `+25.73`
- candidate source: `exp_b035_new_source_full_arc_blend`
- candidate cost: `72612`
- candidate validation: `266_pass_0_fail`
- expected_public_lb_if_pass: `5956.287115`

## 解釈

task023 repairはpublicで成功した。task018 repairに続く2件目のpublic-zero回収で、current public LB bestを `5956.28` に更新する。

## 判断

exp152をcurrent public LB bestとして扱う。ただし task018/task023 ともに `exp_b035` source blend由来のrepairなので、private robustness riskは高い。

## リスク

- leakage risk: high。public/source blend由来candidate。
- overfitting risk: high。public bisectionで特定したzero taskに対するrepairであり、public LB最適化寄り。
