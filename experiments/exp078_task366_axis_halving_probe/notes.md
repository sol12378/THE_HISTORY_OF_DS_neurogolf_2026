# exp078_task366_axis_halving_probe

## 目的

`task366` の片軸半分出力を、2-row/2-col pair reductionで説明できるか調べる。

## 結果

- candidates: `36`
- best: `rows_min` = `0/266`
- mean cell accuracy: `0.2336`

## 判断

full-passがなければ、単純axis halvingでは不足。次はobject mask/source copyを加えたpair compilerへ進む。

## Risk

- leakage risk: low。手書きcandidateのみ。
- overfitting risk: low。採用候補なしなら提出しない。
