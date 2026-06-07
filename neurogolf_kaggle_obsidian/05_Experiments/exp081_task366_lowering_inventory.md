# exp081_task366_lowering_inventory

## 目的

`exp080` のtask366 full-pass ruleをONNX化する前に、static template detectorで小さく表現できるか棚卸しする。

## 結果

- examples: `266`
- unique source templates: `695`
- common used templates: `511`
- max source objects/example: `3`
- max used objects/example: `3`
- max marker cells: `7`
- max patch cells: `49`
- estimated total used kernel params: `7014`

## 判断

naive static template列挙は大きすぎる。hidden汎化の面でも、695 templateの列挙はlookup-likeになる。
一方で、各例のsource object数とmarker数は小さいので、次はobject/marker countの小ささを利用した構造loweringへ進む。

## Lowering Direction

- panel splitはshapeから静的に判定できる。
- source objectは最大3個なので、full-grid connected component unrollではなく、marker位置から必要objectだけを選ぶ。
- marker cellsは最大7個なので、marker sparse coordinatesを使う小さいcandidate placement探索を目指す。
- dynamic component extractionをそのままONNX化しない。過去のflood-fill/large ScatterND失敗パターンに近い。

## Risk

- leakage risk: low-to-medium。all arc-genでcompiler規模を測ったが、raw lookupは生成していない。
- overfitting risk: medium。template列挙は採用しない。
