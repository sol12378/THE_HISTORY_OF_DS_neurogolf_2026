# exp256_current_rank31_80_fullarc_bypass_sweep

## 目的

exp254の外側、current cost rank 31-80 に対して1-pass bypass surgeryをfull-arc gate付きで試す。

## 結果

- generated_candidate_count: `1834`
- sample20_improved_count: `59`
- fullarc_selected_count: `16`
- selected_tasks: `[383, 281, 264, 284, 370, 208, 25, 77, 255, 191, 379, 340, 51, 243, 209, 101]`
- local_delta: `0.3314725494535331`

## 判断

selectedがあればexp255の採点結果を見ながらbundle候補化する。
