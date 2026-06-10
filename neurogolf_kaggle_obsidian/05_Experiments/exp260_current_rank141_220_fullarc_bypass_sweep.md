# exp260_current_rank141_220_fullarc_bypass_sweep

## 目的

exp259 current best の cost rank 141-220 に対して、1-pass bypass surgery を full-arc gate 付きで試す。

## 結果

- base: `experiments/exp259_exp258_bypass_bundle_submit_probe/submission.zip`
- rank_window: `141-220`
- generated_candidate_count: `2529`
- sample20_improved_count: `117`
- fullarc_selected_count: `27`
- local_delta: `+0.8400867054431931`
- selected_tasks: `392, 232, 178, 245, 343, 96, 55, 86, 310, 228, 289, 390, 163, 82, 287, 74, 308, 371, 256, 397, 168, 59, 341, 319, 119, 58, 374`

## 判断

bundle候補化する。rank141-220でもまだ大きなfull-arc安全deltaが残っており、exp255/257/259でLB転写済みのgraph surgery laneとして提出価値がある。

## リスク

- leakage risk: low-medium。既存artifactの構造的なnode bypassであり、source blendよりは低い。
- overfitting risk: medium-low。full-arc gateは通過しているが、private robustnessはLB較正で確認する。
