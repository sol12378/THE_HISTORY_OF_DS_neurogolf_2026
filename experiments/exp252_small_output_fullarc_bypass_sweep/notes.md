# exp252_small_output_fullarc_bypass_sweep

## 目的

small-output高cost候補に対し、既存artifactの1-pass node bypass surgeryをfull-arc gate付きで試し、即提出可能なcost改善があるか確認する。

## 結果

- targets: `[185, 253, 22, 271, 79, 242, 130, 134, 355, 153, 391, 346, 274, 263]`
- generated_candidate_count: `517`
- sample20_improved_count: `2`
- fullarc_selected_count: `1`
- selected_tasks: `[153]`

## 判断

selectedがあればbundle/submission候補。なければこの範囲のquick bypass laneは不発。
