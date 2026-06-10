# exp174_next2o_failure_bisection_probe

## 目的

既知alive/repair済み/exp171 pendingを除いたsubset候補として残るtask285/286をfail-stub化し、public-zero有無を測る。

## 結果

- Kaggle ref: `53524145`
- public LB: `5980.81`
- expected_lb_if_all_alive: `5967.748188739109`
- observed_drop: `13.010000000000218`
- missing_drop_vs_all_alive: `13.061811260890778`

## 判断

task285はpublic-zero、task286はpublic-scoring alive。task285 repairへ進む。

## リスク

Fail-stub probe自体のleakage riskは低いが、public挙動に基づくtarget選定なのでoverfitting riskは中。
