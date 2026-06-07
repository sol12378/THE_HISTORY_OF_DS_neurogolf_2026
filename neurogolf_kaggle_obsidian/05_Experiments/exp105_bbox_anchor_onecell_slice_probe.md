# exp105_bbox_anchor_onecell_slice_probe

## Hypothesis

exp104で絶対座標static index mapが否定されたため、1x1 P0 output taskなら、nonzero bbox top-leftを `ReduceSum + ArgMax` で求め、固定offsetの1セルを `Slice + Pad` する `tiny_dynamic_shape_index` candidateが成立する可能性がある。

## Result

- campaign #: `7`
- base local estimate: `6282.812218`
- targets: `4`
- bbox-min offset fits: `0`
- generated candidates: `0`
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.812218`
- submission decision: `no_submit`

## Interpretation

task355/346/48/291の1x1 P0 taskでは、全nonzero bbox top-left基準の固定offsetでは出力セルを説明できなかった。`tiny_dynamic_shape_index` の方向性自体は残るが、anchorは全nonzero bboxではなく、色別bbox・object role・component relationにする必要がある。

## Risk

- leakage risk: medium。
- overfitting risk: medium。offset ruleは小さいがtask-specificになりやすい。
