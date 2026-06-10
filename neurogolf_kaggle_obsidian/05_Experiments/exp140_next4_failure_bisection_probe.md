# exp140_next4_failure_bisection_probe

## 目的

exp139でtop4 `013/002/029/009` がpublic-scoring aliveと分かったため、次点group `024/018/004/019` を fail-stub probe で測る。

## 方法

exp127 best zipをベースに、task024/018/004/019 だけ `output = input * 0` の fail stub に差し替えた。対象4 taskはローカルで全て `0_pass_1_fail`。

## 結果

- Kaggle ref: `53520700`
- public LB: `5886.89`
- exp127 LB `5930.55` からのdrop: `43.66`
- 4 taskが全て生きていた場合の期待drop: `57.01524`
- 差分: `-13.35524`
- task018 の点数: `13.35099`

## 解釈

observed drop は task024 + task004 + task019 の点数和と一致し、task018 の点数分だけ不足した。したがって、task018 は exp127 時点で既に public scoring 0点だった可能性が非常に高い。

## 判断

task018 を最優先repair対象にする。修復できれば LB `+13.35` 級を回収できる可能性がある。

## リスク

- leakage risk: low。
- overfitting risk: medium。public 0点推定は強いが、修復候補は full validation と小規模LB較正が必要。
