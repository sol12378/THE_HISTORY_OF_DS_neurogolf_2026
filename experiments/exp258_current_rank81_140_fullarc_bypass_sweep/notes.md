# exp258_current_rank81_140_fullarc_bypass_sweep

## 目的

exp257 current bestのcost rank 81-140 に対して1-pass bypass surgeryをfull-arc gate付きで試す。

## 結果

- generated_candidate_count: `1980`
- sample20_improved_count: `77`
- fullarc_selected_count: `23`
- selected_tasks: `[233, 177, 118, 330, 97, 367, 89, 324, 182, 224, 325, 80, 14, 378, 184, 231, 252, 27, 65, 46, 368, 196, 288]`
- local_delta: `0.21905617266236455`

## 判断

selectedがあればbundle候補化する。exp255/257でLB転写済みのため、十分なdeltaなら提出優先。
