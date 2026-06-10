# exp260_current_rank141_220_fullarc_bypass_sweep

## 目的

exp259 current bestのcost rank 141-220 に対して1-pass bypass surgeryをfull-arc gate付きで試す。

## 結果

- generated_candidate_count: `2529`
- sample20_improved_count: `117`
- fullarc_selected_count: `27`
- selected_tasks: `[392, 232, 178, 245, 343, 96, 55, 86, 310, 228, 289, 390, 163, 82, 287, 74, 308, 371, 256, 397, 168, 59, 341, 319, 119, 58, 374]`
- local_delta: `0.8400867054431931`

## 判断

selectedがあればbundle候補化する。exp255/257/259でLB転写済みのため、十分なdeltaなら提出優先。
