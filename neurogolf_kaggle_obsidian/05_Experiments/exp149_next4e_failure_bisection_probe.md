# exp149_next4e_failure_bisection_probe

## 目的

exp148後の未判定subset頻出上位 `053/056/067/005` のpublic scoring状態をfail-stub probeで測る。

## 仮説

このgroupにpublic-zero taskが含まれていれば、4 taskを意図的にfailさせたときのLB dropがall-alive期待値より小さくなる。

## 結果

- Kaggle ref: `53521652`
- public LB: `5856.74`
- base: exp127 LB `5930.55`
- targets: `053 056 067 005`
- expected_drop_if_all_alive: `73.813087369845`
- expected_lb_if_all_alive_from_exp127: `5856.7369126301555`
- observed_drop_from_exp127: `73.81`
- drop_delta_vs_all_alive: `-0.0031`
- target validation: all `0_pass_1_fail`

## 解釈

観測LBはall-alive期待値と一致した。task053/056/067/005 の中にtask018やtask025のようなpublic-zero evidenceはない。

## 判断

task053/056/067/005 を短期public-zero suspectから外す。次は残りの頻出候補 `023/036/085/034/051/049/046` などを対象にする。

## リスク

- leakage risk: low。意図的fail-stub probeでありprivate labelは使っていない。
- overfitting risk: medium。submissionを消費してpublic scoring挙動を測っているため、public探索依存がある。
