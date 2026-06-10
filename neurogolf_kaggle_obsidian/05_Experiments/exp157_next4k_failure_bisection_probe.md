# exp157_next4k_failure_bisection_probe

## 目的

exp156採点待ち中に準備した、未判定subset候補 `090/173/028/091` のpublic scoring状態をfail-stub probeで測る。

## 結果

- Kaggle ref: `53522643`
- public LB: `5873.63`
- base: exp127 LB `5930.55`
- targets: `090 173 028 091`
- expected_drop_if_all_alive: `56.917033011886396`
- expected_lb_if_all_alive_from_exp127: `5873.632966988113`
- observed_drop_from_exp127: `56.92`
- drop_delta_vs_all_alive: `0.0030`
- target validation: all `0_pass_1_fail`

## 解釈

観測LBはall-alive期待値と一致した。task090/173/028/091 の中にpublic-zero evidenceはない。

## 判断

task090/173/028/091 を短期public-zero suspectから外す。subset頻出候補は薄くなっており、追加bisectionの期待値は下がった。次は残り候補を低優先で保持しつつ、task025 rule repairやPhase C dtype/cost laneへの切替を検討する。

## リスク

- leakage risk: low。意図的fail-stub probeでありprivate labelは使っていない。
- overfitting risk: medium。submissionを消費してpublic scoring挙動を測っているため、public探索依存がある。
