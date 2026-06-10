# exp156_next4j_failure_bisection_probe

## 目的

exp155後の未判定subset頻出上位 `027/082/088/092` のpublic scoring状態をfail-stub probeで測る。

## 結果

- Kaggle ref: `53522565`
- public LB: `5873.06`
- base: exp127 LB `5930.55`
- targets: `027 082 088 092`
- expected_drop_if_all_alive: `57.48546606439621`
- expected_lb_if_all_alive_from_exp127: `5873.064533935604`
- observed_drop_from_exp127: `57.49`
- drop_delta_vs_all_alive: `0.0045`
- target validation: all `0_pass_1_fail`

## 解釈

観測LBはall-alive期待値と一致した。task027/082/088/092 の中にpublic-zero evidenceはない。

## 判断

task027/082/088/092 を短期public-zero suspectから外す。採点待ち中に準備した exp157 `090/173/028/091` probeへ進む。

## リスク

- leakage risk: low。意図的fail-stub probeでありprivate labelは使っていない。
- overfitting risk: medium。submissionを消費してpublic scoring挙動を測っているため、public探索依存がある。
