# exp_b011_task037_diag_ray_lowering

## 目的

exp_b010でfull passしたtask037 `opposite_ray_diag_same_color` ruleを実ONNXへloweringし、strict seedより低costになるかを確認する。

## 結果

- strict baseline task cost: `63726`
- attempted kernels: `19, 21, 23, 25, 27, 29`
- validation: all candidates `266_pass_0_fail`
- best candidate cost: `441976`
- status: `no_gain`
- local estimate delta: `0.0`
- submission: `no_submit`

## 解釈

ruleは正しいが、depthwise Convで対角方向の可視性を作るloweringは高すぎる。正しいruleを見つけることと、NeuroGolfで安いONNXに落とすことは別問題である。

## Risk

- leakage risk: low。説明可能rule。
- overfitting risk: medium。ruleは全arc-gen通過だが、採用候補ではない。

## Decision

full-grid Conv visibilityをguardrail入り。次はdiagonal-specific small mask、triangular Gather、または既存artifact surgeryでtask037を<=600、理想<=250へ落とす。
