# exp256_current_rank31_80_fullarc_bypass_sweep

## 目的

exp254の外側、current cost rank31-80に対して1-pass bypass surgeryをfull-arc gate付きで探索する。

## 結果

- base_bundle: `experiments/exp234_task300_max_color_submit_probe/submission.zip`
- generated_candidate_count: `1834`
- sample20_improved_count: `59`
- fullarc_selected_count: `16`
- selected_tasks: `[383, 281, 264, 284, 370, 208, 25, 77, 255, 191, 379, 340, 51, 243, 209, 101]`
- local_delta: `+0.3314725494535331`
- 最大ヒット: task340 `Cast_node0_to_input0`, cost `70342 -> 52342`, local `+0.29557`

## 判断

exp255でbundle surgeryがLB転写することを確認できたため、exp257で16件をexp255へ積んで提出する。

## リスク

- leakage risk: low-medium。既存artifactのfull-arc gated surgery。
- overfitting risk: medium-low。available examples全通過だが、private robustnessはLB確認待ち。
