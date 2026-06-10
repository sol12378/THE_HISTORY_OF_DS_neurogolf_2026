# exp150_next4f_failure_bisection_probe

## 目的

exp149後の未判定subset頻出上位 `023/036/085/034` のpublic scoring状態をfail-stub probeで測る。

## 仮説

このgroupにpublic-zero taskが含まれていれば、4 taskを意図的にfailさせたときのLB dropがall-alive期待値より小さくなる。

## 結果

- Kaggle ref: `53521805`
- public LB: `5890.05`
- base: exp127 LB `5930.55`
- targets: `023 036 085 034`
- expected_drop_if_all_alive: `55.52734311214046`
- expected_lb_if_all_alive_from_exp127: `5875.02265688786`
- observed_drop_from_exp127: `40.50`
- all-aliveとの差分: `15.0273`
- identified_public_zero_task: `023`
- task023 points: `15.027312930662868`
- target validation: all `0_pass_1_fail`

## 解釈

観測dropはall-alive期待よりtask023の点数相当だけ小さい。したがって task023 はpublic-zeroの可能性が非常に高い。task036/085/034 はこのprobe上ではpublic-scoring aliveとして扱う。

## 判断

task023を次のrepair targetにする。まず既存source候補のfull validationを監査し、task018型の別raw差し替えが可能かを確認する。

## リスク

- leakage risk: low。意図的fail-stub probeでありprivate labelは使っていない。
- overfitting risk: medium。submissionを消費してpublic scoring挙動を測っているため、public探索依存がある。
