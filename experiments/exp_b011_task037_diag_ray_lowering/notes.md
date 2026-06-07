# exp_b011_task037_diag_ray_lowering

## 目的

exp_b010でfull passしたtask037 `opposite_ray_diag_same_color` ruleを実ONNXへloweringし、strict seedより低costになるかを確認する。

## 結果

- baseline task cost: 63726
- baseline task points: 13.937652
- accepted: None
- local estimate delta: 0.000000
- new strict seed local estimate: 6282.230228

## 解釈

対角方向の同色端点に挟まれたbackgroundを埋めるruleはlocal all arc-genで正しい。Conv可視性loweringがcost gainを出せば、単一task delta submissionでlocal/LB対応を見る。

## Risk

- leakage risk: low。説明可能ruleで、signature lookupではない。
- overfitting risk: medium。arc-gen full pass済みでもhidden生成差をKaggle deltaで確認する。

## Decision

acceptedがあればsingle-task delta calibrationとして提出候補。なければ、Convではなくdiagonal-specific small reductionへloweringを縮小する。
