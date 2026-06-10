# exp195_channel_gather_colormap_miner

## 目的

C-2 `channel_gather_colormap` archetypeを全taskへ適用し、full-validかつcost改善する1-node Gather候補を探す。

## 結果

- mapping_ok_count: `2`
- improved_count: `0`
- improved_tasks: `[]`
- submission_decision: `no_submit_no_improved_candidates`

## リスク

low: input-only color permutation rule, no private labels.
low-to-medium: full arc-gen validation required, but public/private robustness still needs LB calibration if submitted.
