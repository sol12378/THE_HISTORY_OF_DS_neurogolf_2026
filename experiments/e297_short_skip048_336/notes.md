# e297_short_skip048_336

## 目的

exp261 current bestのcost rank 221-320 に対して1-pass bypass surgeryをfull-arc gate付きで試す。

## 結果

- generated_candidate_count: `2569`
- score_failure_count: `0`
- sample20_improved_count: `64`
- fullarc_selected_count: `20`
- selected_tasks: `[30, 345, 124, 78, 153, 188, 329, 212, 50, 3, 254, 369, 180, 45, 248, 357, 273, 60, 226, 123]`
- local_delta: `0.6051077060010481`

## 判断

selectedがあればbundle候補化する。deltaが小さい場合は次windowや低cost lowering laneを優先する。
