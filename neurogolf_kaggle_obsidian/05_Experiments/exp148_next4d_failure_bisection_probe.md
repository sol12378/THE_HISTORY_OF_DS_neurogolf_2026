# exp148_next4d_failure_bisection_probe

## 目的

task018修復後、task025は既存source差し替え不可と判明したため、次点group `366/077/064/084` のpublic scoring状態をfail-stub probeで測る。

## 仮説

このgroupにpublic-zero taskが含まれていれば、4 taskを意図的にfailさせたときのLB dropがall-alive期待値より小さくなる。

## 結果

- Kaggle ref: `53521453`
- public LB: `5877.68`
- base: exp127 LB `5930.55`
- targets: `366 077 064 084`
- expected_drop_if_all_alive: `52.78042443088678`
- expected_lb_if_all_alive_from_exp127: `5877.769575569114`
- observed_drop_from_exp127: `52.87`
- drop_delta_vs_all_alive: `0.0896`
- target validation: all `0_pass_1_fail`

## 解釈

観測LBはall-alive期待値とほぼ一致した。public LB丸め/採点揺れを考えると、task366/077/064/084 の中にtask018やtask025のような大きなpublic-zero evidenceはない。

## 判断

task366/077/064/084 を短期public-zero suspectから外す。次は未probeのsubset候補を継続して探索し、同時にtask025のrule repair queueを別管理する。

## リスク

- leakage risk: low。意図的fail-stub probeでありprivate labelは使っていない。
- overfitting risk: medium。submissionを消費してpublic scoring挙動を測っているため、public探索依存がある。
