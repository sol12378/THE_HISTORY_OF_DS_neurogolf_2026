# exp258_current_rank81_140_fullarc_bypass_sweep

## 目的

exp257 current bestのcost rank81-140に対して1-pass bypass surgeryをfull-arc gate付きで探索する。

## 結果

- base_bundle: `experiments/exp257_exp256_bypass_bundle_submit_probe/submission.zip`
- generated_candidate_count: `1980`
- sample20_improved_count: `77`
- fullarc_selected_count: `23`
- selected_tasks: `[233, 177, 118, 330, 97, 367, 89, 324, 182, 224, 325, 80, 14, 378, 184, 231, 252, 27, 65, 46, 368, 196, 288]`
- local_delta: `+0.21905617266236455`

## 判断

exp255/257で同種bundleのLB転写が確認済み。23件をexp259でbundle提出する。

## リスク

- leakage risk: low-medium。既存artifactのfull-arc gated surgery。
- overfitting risk: medium-low。available examples全通過だが、private robustnessはLB確認待ち。
