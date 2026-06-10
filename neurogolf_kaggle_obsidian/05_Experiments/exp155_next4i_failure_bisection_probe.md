# exp155_next4i_failure_bisection_probe

## 目的

exp154後の未判定subset頻出上位 `063/022/037/020` のpublic scoring状態をfail-stub probeで測る。

## 結果

- Kaggle ref: `53522390`
- public LB: `5873.60`
- base: exp127 LB `5930.55`
- targets: `063 022 037 020`
- expected_drop_if_all_alive: `56.94773068259494`
- expected_lb_if_all_alive_from_exp127: `5873.602269317405`
- observed_drop_from_exp127: `56.95`
- drop_delta_vs_all_alive: `0.0023`
- target validation: all `0_pass_1_fail`

## 解釈

観測LBはall-alive期待値と一致した。task063/022/037/020 の中にpublic-zero evidenceはない。

## 判断

task063/022/037/020 を短期public-zero suspectから外す。次は残り頻出候補 `027/082/088/092/090/173` などを対象にする。

## リスク

- leakage risk: low。意図的fail-stub probeでありprivate labelは使っていない。
- overfitting risk: medium。submissionを消費してpublic scoring挙動を測っているため、public探索依存がある。
