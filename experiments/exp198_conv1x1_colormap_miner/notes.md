# exp198_conv1x1_colormap_miner

## 目的

C-2 `one_node_conv_kernel` の最小形として、非injectiveも許す1x1 Conv色写像を全taskへ適用する。

## 結果

- mapping_ok_count: `4`
- improved_count: `0`
- improved_tasks: `[]`
- submission_decision: `no_submit_no_improved_candidates`

## リスク

low: input-only per-color mapping, no private labels.
low-to-medium: full arc-gen validation required, but private robustness still needs LB calibration if submitted.
