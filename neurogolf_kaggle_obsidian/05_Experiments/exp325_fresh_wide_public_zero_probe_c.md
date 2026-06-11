# exp325 fresh wide public-zero probe C

## 目的

exp323 all-alive 後の fresh wide public-zero probe。既診断 task を除外し、残る `franksunp_blended_best` accepted 高リスク task を16件 fail-stub する。

## 仮説

未特定 public-zero が残っているなら、未診断 public-code 高リスク task を fail-stub した時に observed drop が expected all-alive drop より小さくなる。

## 結果

- targets: task334/373/380/236/106/266/026/318/144/267/052/083/108/142/152/194
- target validation: all `0_pass_1_fail`
- expected all-alive drop: `289.001636451155`
- expected all-alive LB: `5719.958363548845`
- Kaggle ref: `53552426`
- Public LB: `5719.95`

## 判断

Public LB は expected all-alive LB と scoreboard rounding 内で一致した。今回の16 task は public-scoring alive と扱い、immediate public-zero repair suspect から除外する。

exp323/325 と連続で all-alive になったため、`franksunp_blended_best` 高point順の wide probe 収率は低い。次は強い新 suspect source が無い限り、GridSample/task037 lowering へ主軸を移す。

## Risk

- leakage risk: low。意図的 fail-stub 診断のみ。
- overfitting risk: medium。public diagnostic なので、missing drop が出た場合も split/repair で確認する必要がある。

## Next

task037 の bounded diagonal shift / GridSample equality mask lowering を設計する。
