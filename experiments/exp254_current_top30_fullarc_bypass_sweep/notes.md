# exp254_current_top30_fullarc_bypass_sweep

## 目的

current best exp234 bundle上のtop30 cost taskに対し、現rawのcostを測り直したうえで1-pass bypass surgeryをfull-arc gate付きで試す。

## 結果

- generated_candidate_count: `1634`
- sample20_improved_count: `66`
- fullarc_selected_count: `13`
- selected_tasks: `[133, 204, 216, 54, 280, 205, 382, 239, 398, 275, 158, 234, 19]`
- local_delta: `0.09024144378056853`

## 判断

selectedがあればbundle/submission候補。なければtop30 quick bypassは不発。
