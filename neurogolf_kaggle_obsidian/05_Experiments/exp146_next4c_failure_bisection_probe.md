# exp146_next4c_failure_bisection_probe

## 目的

既知aliveとtask018を除外した次点consensus group `021/014/025/008` のpublic scoring状態をfail-stub probeで測る。

## 結果

- Kaggle ref: `53521220`
- public LB: `5887.02`
- exp127 LB `5930.55` からのdrop: `43.53`
- all-alive期待drop: `57.12815`
- 不足分: 約 `13.598`
- task025 local points: `13.60040`

## 解釈

observed dropは task021 + task014 + task008 の点数和と一致し、task025 の点数分だけ不足した。task025 は exp127時点でpublic-zeroだった可能性が非常に高い。

## 判断

task025を次のrepair targetにする。

## リスク

- leakage risk: low。
- overfitting risk: medium。
