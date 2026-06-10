# exp254_current_top30_fullarc_bypass_sweep

## 目的

current best exp234 bundle上のtop30 cost taskを測り直し、1-pass bypass surgeryをfull-arc gate付きで探索する。

## 結果

- base_bundle: `experiments/exp234_task300_max_color_submit_probe/submission.zip`
- targets: top30 current-cost tasks
- generated_candidate_count: `1634`
- sample20_improved_count: `66`
- fullarc_selected_count: `13`
- selected_tasks: `[133, 204, 216, 54, 280, 205, 382, 239, 398, 275, 158, 234, 19]`
- local_delta: `+0.09024144378056853`

## 判断

単発では小さいが、13 task bundleならLB表示差分が期待できる。exp255で提出候補化する。

## リスク

- leakage risk: low-medium。既存artifactのfull-arc gated graph surgeryであり、raw source差し替えではない。
- overfitting risk: medium-low。available examples全通過だが、private robustnessはLB較正が必要。
