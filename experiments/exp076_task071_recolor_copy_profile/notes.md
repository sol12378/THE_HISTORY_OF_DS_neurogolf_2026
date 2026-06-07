# exp076_task071_recolor_copy_profile

## 目的

`task071` のmulti-color recolor/copyを、低cost ONNXに落とせる小branch tree候補へ分解する。

## 仮説

変更セルの出力色またはzero maskは、component-local座標、入力色rank、4近傍色から高純度に決まる。

## 結果

- changed cells: `4260`
- best out_color feature: `dr+dc+h+w+in_color+nbr4` accuracy `0.9192` keys `3194` ambiguous `279`
- best zero-mask feature: `dr+dc+h+w+in_color+nbr4` accuracy `0.9399` keys `3194` ambiguous `234`

## 判断

この実験は診断のみで、提出候補は生成していない。
出力色featureが低純度なら、`task071` は直接の小ONNX化ではなく、zero maskとsource color copy directionを分けたcompilerへ回す。

## Risk

- leakage risk: low。全arc-genを使う診断だが、raw key lookupを採用していない。
- overfitting risk: medium。高純度でもkey数が多いfeatureはbranch圧縮なしでは採用しない。
