# exp196_static_shift_slice_pad_miner

## 結果

- shift_ok_count: `1`
- improved_count: `0`
- submission_decision: `no_submit_no_improved_candidates`

## 判断

固定平行移動 + zero padding の `Slice+Pad` 横展開では改善なし。次は `one_node_conv_kernel` またはtask-specific loweringへ移る。

## リスク

input-only固定変換なのでleakage riskは低い。
