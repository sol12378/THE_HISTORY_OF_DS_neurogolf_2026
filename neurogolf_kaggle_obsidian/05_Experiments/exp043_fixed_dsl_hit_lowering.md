# exp043_fixed_dsl_hit_lowering

## 目的

exp042のproxy hitのうち固定shapeで安全にONNX化できる変換を、実costで検証する。

## 結果

- status: no_gain
- baseline local estimate: 6480.302478
- new local estimate: 6480.302478
- delta: +0.000000
- candidate status: skipped 4, rejected 1, no_cost_gain 4

## 解釈

固定rot/crop/upscaleは変換としては正しいが、既存artifactがすでにかなり安い。naiveな `Slice/Gather/Pad/Tile` loweringでは勝てない。

## 判断

単純DSLをそのままONNX化する方針は弱い。DSL探索は正しいが、必要なのは「意味の発見」ではなく、既存artifactより安い専用loweringである。

## 次

variable-shape flip と object/bbox 系に移る。ただし大きなindex initializerやfull-grid操作は事前にguardrailへ入れる。
