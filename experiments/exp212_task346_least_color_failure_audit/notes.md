# exp212_task346_least_color_failure_audit

## 目的

task346で`least_nz`が`263/267`まで通るため、残り4failがtie-breakで解けるか確認する。

## 結果

- least_nz_pass: `263/267`
- fail_count: `4`
- output_rank_hist: `{0: 263, 1: 4}`

## 判断

failがcount tie由来ならnon-lookup tie-breakerを追加する。そうでなければ1x1 aggregate laneからpivotする。
