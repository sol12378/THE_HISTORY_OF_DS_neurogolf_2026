# exp044_task031_dynamic_bbox_lowering

## 仮説

task031の `bbox_nonzero` はexp042最大hitであり、動的bboxを `ArgMax + GatherND` で閉形式loweringすれば、既存artifact cost 18616を下回れる可能性がある。

## 実験

- 対象: task031
- rule: nonzero bboxを左上へcrop
- lowering: `ReduceSum/ReduceMax/ArgMax` でbboxを検出し、`GatherND` で全channelを左上へshift、bbox外をmaskする
- validation: train + test + all arc-gen

## 結果

- status: no_cost_gain
- baseline cost: 18616
- candidate cost: 1050228
- baseline local estimate: 6480.302478
- new local estimate: 6480.302478
- delta: 0.000000
- reason: candidate cost is not lower than baseline

## 解釈

この実験はobject/bbox compilerの最初の実cost検証である。改善しない場合、`GatherND` 用の巨大indexテンソルがcostを支配している可能性が高く、bbox専用のより小さいloweringが必要。

## リスク

- leakage risk: 低。signature lookupなし。
- overfitting risk: 低〜中。all arc-gen通過時のみ採用。

## 次

改善しなければ、`GatherND` full-grid方式をguardrailへ入れ、row/colごとのSlice候補選択や既存artifact surgeryとのhybridへ移る。
