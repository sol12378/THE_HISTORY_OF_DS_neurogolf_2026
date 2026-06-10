# exp264_current_rank321_400_fullarc_bypass_sweep

## 目的

exp263 current bestのcost rank 321-400 に対して1-pass bypass surgeryをfull-arc gate付きで試す。

## 結果

- generated_candidate_count: `500`
- score_failure_count: `0`
- sample20_improved_count: `11`
- fullarc_selected_count: `8`
- selected_tasks: `[299, 229, 235, 339, 129, 144, 318, 26]`
- local_delta: `0.5976596441240858`

## 判断

selectedがあればreplay bundle候補化する。提出はreplay後にlocal estimateが更新された場合のみ。
