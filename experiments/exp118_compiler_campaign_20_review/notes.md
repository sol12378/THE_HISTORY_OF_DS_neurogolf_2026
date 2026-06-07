# exp118_compiler_campaign_20_review

## Purpose

compiler campaign #20の必須local estimate測定と、#16〜#20のLB有意義性レビュー。

## Result

- local delta #1〜#20: `+0.1239388580395987`
- local delta #16〜#20: `0.000000`
- current submit-safe working local: `6282.93615685804`

## Interpretation

#16〜#20はlocal gainを出さなかった。ただしtask185について、次の判断材料を得た。

- homogeneous-2x2 coreはfinal Pad込みcost `1516` まで下がる。
- 固定lattice座標はfull validation `0_pass_1_fail`。
- input-only window selectorはPythonで `267/267`。
- direct ONNX selector plumbingは高cost: line detection `4204`、branchless spacing core `5160`、GatherND/ArgMax scoring `76194`。

したがって、task185のdirect dynamic ONNXは現時点の主score laneではない。

## Next Policy

#21〜#25はtask366/task365など、別のsolved high-gain ruleにpivotする。task185は、`GatherND/ArgMax` やfull-grid line detectionを避けるfused selector案が出た場合だけ戻る。

## Risk

- leakage risk: low。
- overfitting risk: medium。task185をraw position tableで提出しない。
