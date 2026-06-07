# exp072_task251_predicate_probe

## 目的

`exp071` で最優先候補になった `task251` について、単色fillを低複雑度changed-cell predicateでどこまで説明できるか確認する。

## 結果

- `inside_ray4_adj1plus`: TP `2708`, FP `113`, FN `0`, TN `20121`
- `inside_ray4_adj2`: TP `2170`, FP `111`, FN `538`
- `reject_straight_adj2`: TP `2006`, FP `4`, FN `702`

## 解釈

`inside nonzero bbox + 4方向rayで非ゼロに挟まれる + 隣接非ゼロあり` は全positiveを拾える。これはtask251の合成問題を「113 false positiveをどう削るか」に縮小した。

ただし、直線接触を単純にrejectするとFNが702出る。FP signatureもsingleton寄りなので、raw signature tableではなく、ray distance/local component contextを使う小branch treeが必要。

## 次アクション

深さ2〜4のbranch tree synthesizerを作り、base predicate内のFPだけを削る。full arc-gen passした場合のみONNX loweringへ進み、single-task deltaでLB較正する。
