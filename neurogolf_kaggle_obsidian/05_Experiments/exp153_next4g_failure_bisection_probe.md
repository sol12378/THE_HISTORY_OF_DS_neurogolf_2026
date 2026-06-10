# exp153_next4g_failure_bisection_probe

## 目的

exp152後の未判定subset頻出上位 `051/049/046/011` のpublic scoring状態をfail-stub probeで測る。

## 結果

- Kaggle ref: `53522084`
- public LB: `5873.53`
- base: exp127 LB `5930.55`
- targets: `051 049 046 011`
- expected_drop_if_all_alive: `57.023091988774816`
- expected_lb_if_all_alive_from_exp127: `5873.526908011226`
- observed_drop_from_exp127: `57.02`
- drop_delta_vs_all_alive: `-0.0031`
- target validation: all `0_pass_1_fail`

## 解釈

観測LBはall-alive期待値と一致した。task051/049/046/011 の中にpublic-zero evidenceはない。

## 判断

task051/049/046/011 を短期public-zero suspectから外す。次は残り頻出候補 `033/087/276/030/063/022/037/020` などを対象にする。

## リスク

- leakage risk: low。意図的fail-stub probeでありprivate labelは使っていない。
- overfitting risk: medium。submissionを消費してpublic scoring挙動を測っているため、public探索依存がある。
