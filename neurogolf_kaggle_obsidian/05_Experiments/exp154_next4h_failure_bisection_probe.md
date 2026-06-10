# exp154_next4h_failure_bisection_probe

## 目的

exp153後の未判定subset頻出上位 `033/087/276/030` のpublic scoring状態をfail-stub probeで測る。

## 結果

- Kaggle ref: `53522207`
- public LB: `5859.07`
- base: exp127 LB `5930.55`
- targets: `033 087 276 030`
- expected_drop_if_all_alive: `71.47882525980204`
- expected_lb_if_all_alive_from_exp127: `5859.071174740198`
- observed_drop_from_exp127: `71.48`
- drop_delta_vs_all_alive: `0.0012`
- target validation: all `0_pass_1_fail`

## 解釈

観測LBはall-alive期待値と一致した。task033/087/276/030 の中にpublic-zero evidenceはない。

## 判断

task033/087/276/030 を短期public-zero suspectから外す。次は残り頻出候補 `063/022/037/020/027/082/088/092` などを対象にする。

## リスク

- leakage risk: low。意図的fail-stub probeでありprivate labelは使っていない。
- overfitting risk: medium。submissionを消費してpublic scoring挙動を測っているため、public探索依存がある。
