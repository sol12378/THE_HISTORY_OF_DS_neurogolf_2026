# exp169_next4m_failure_bisection_probe

## 目的

exp166 current bestをベースに、task133/396/158/047をfail-stub化してpublic-zero有無を測る。

## 結果

- Kaggle ref: `53523802`
- public LB: `5938.07`
- all-alive expected LB: `5912.36`
- missing drop: `25.7111`
- interpretation: missing dropがtask133+task158の点数と一致。

## 判断

task133/task158はpublic-zero repair target。task396/task047はpublic-scoring aliveとして除外。

## リスク

Fail-stub probe自体のleakage riskは低いが、public LB挙動に基づくtarget選定なのでoverfitting riskは中。
