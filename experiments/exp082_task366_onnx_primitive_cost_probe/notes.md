# exp082_task366_onnx_primitive_cost_probe

## 目的

`task366` object-marker copy loweringに使うONNX primitiveの公式costを測る。

## 結果

- primitive count: `5`
- scored count: `5`

## 判断

これは提出候補ではない。`Slice` / `Pad` / `Conv` / `Where` の実costを見て、次のcorrectness-first compilerのpre-emission budgetを決める。

## Risk

- leakage risk: low。task labelを埋め込んでいない。
- overfitting risk: low。cost-only診断。
