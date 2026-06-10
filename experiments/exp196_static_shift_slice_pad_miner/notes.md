# exp196_static_shift_slice_pad_miner

## 目的

C-2 `static_slice_pad` archetypeの最小形として、固定平行移動+zero paddingを全taskへ適用する。

## 結果

- shift_ok_count: `1`
- improved_count: `0`
- improved_tasks: `[]`
- submission_decision: `no_submit_no_improved_candidates`

## リスク

low: input-only fixed shift rule, no private labels.
low-to-medium: full arc-gen validation required, but private robustness still needs LB calibration if submitted.
