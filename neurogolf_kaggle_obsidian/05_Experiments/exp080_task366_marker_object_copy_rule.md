# exp080_task366_marker_object_copy_rule

## 目的

`task366` をsource panel object と target panel marker の対応として説明できるか検証する。

## 結果

- candidate: `object_marker_copy_by_color_shape`
- validation: `266/266`
- train: `3/3`
- test: `1/1`
- arc-gen: `262/262`
- mean cell accuracy: `1.0`

## Rule

入力をsource object panelとtarget marker panelへ分ける。source panelの非背景objectを抽出し、target panel上のmarker色/marker形状に対応するsource objectだけをtarget背景へ貼り付ける。targetに対応markerがないsource objectはdistractorとしてskipする。

## 判断

task366の正答ruleは見つかった。次はcost-aware ONNX lowering。dynamic connected componentや大きなindex tensorは高costになりやすいため、panel split、marker/object shape matching、pasteを小さな `Slice` / `Gather` / `Where` 系へ分解する。

## Risk

- leakage risk: low-to-medium。raw lookupではなく構造ruleだが、task-specific。
- overfitting risk: medium。ONNX化後にall-arc full validationとcost gateが必要。
