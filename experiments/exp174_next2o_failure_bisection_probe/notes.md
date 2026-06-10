# exp174_next2o_failure_bisection_probe

## 目的

既知alive/repair済み/exp171 pendingを除いたsubset候補として残る `285/286` をfail-stub化し、public-zero有無を測る準備をする。

## 結果

- status: `submitted_complete`
- targets: `[285, 286]`
- expected_drop_if_all_alive: `26.071811260890996`
- expected_lb_if_all_alive: `5967.748188739109`
- zip sha256: `e7944b9325c2e0f7899c09c59390954f550449c44a6b12808916daa263cd4fc8`
- kaggle_ref: `53524145`
- public_lb: `5980.81`
- observed_drop: `13.010000000000218`
- missing_drop_vs_all_alive: `13.061811260890778`
- interpretation: observed dropはtask286 alive分に一致し、missing dropはtask285点数に一致。task285がpublic-zero、task286はalive。

## Target Validation

- task285: `0_pass_1_fail`, reason `mismatch example 0`
- task286: `0_pass_1_fail`, reason `mismatch example 0`

## 判断

task285 repairへ進む。task286は短期public-zero suspectから除外。

## リスク

low: deliberate fail-stub probe; no private labels used.
medium: consumes a submission if used and only measures public scoring behavior for two tasks.
