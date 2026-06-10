# exp145_next4b_failure_bisection_probe

## 目的

task018修復後、残るpublic-zero候補をさらに絞るため、次点consensus group `032/050/016/031` をfail-stub probeで測る。

## 結果

- Kaggle ref: `53521106`
- public LB: `5859.97`
- exp127 LB `5930.55` からのdrop: `70.58`
- all-alive期待drop: `70.57623`
- drop error: 約 `0.00377`

## 解釈

task032/050/016/031 はpublic-scoring alive。これらは public-zero gap の即時repair対象ではない。

## 判断

次groupへ進む。既知alive: `013/002/029/009/024/004/019/032/050/016/031`。public-zero repair済み: `018`。

## リスク

- leakage risk: low。
- overfitting risk: medium。
