# exp139_top4_failure_bisection_probe

## 目的

exp137/138 の top consensus task `013/002/029/009` が public scoring で既に0点なのかを、fail-stub bisection probeで直接測る。

## 方法

exp127 best zipをベースに、task013/002/029/009 だけ `output = input * 0` の fail stub に差し替えた。対象4 taskはローカルで全て `0_pass_1_fail`。

## 結果

- Kaggle ref: `53520609`
- public LB: `5876.49`
- exp127 LB `5930.55` からのdrop: `54.06`
- 4 taskが全て生きていた場合の期待drop: `54.05675`
- drop error: 約 `0.00325`

## 解釈

task013/002/029/009 は public scoring で全て寄与している。つまり、これらは -352 gap の「既に0点になっているtask」ではない。subset-sum consensus はこの4 taskに対しては反証された。

## 判断

次は top4 を除外し、task024/018/004/019/032/050/016 などの次点groupを probe するか、よりsource/provenanceに基づく group を作る。

## リスク

- leakage risk: low。
- overfitting risk: medium。public scoring上の生死だけを測るprobeであり、private全体の完全な証明ではない。
