# exp_b031_task085_horizontal_bar_alternate_erase_rule

## 目的

`exp071` で `object_erase_or_mask_clean` と判定された task085 を説明可能ruleへ圧縮する。

## 結果

- task: `85`
- rule: `horizontal_3row_bar_middle_alternate_erase`
- validation: `265/265`
- train: `2/2`
- test: `1/1`
- arc-gen: `262/262`
- avg changed cells: `19.5585`
- max changed cells: `55`

## Rule

同一色で構成された height `3` のsolid horizontal rectangleを見つけ、その中央行だけ、component左端から奇数offsetのセルを `0` へ消す。

## Lowering Note

global checkerboard parityではなく、component left edgeからのrelative parityが必要。既存artifactはConv/Floor/Subでこの相対parityを実装している可能性が高い。

## Decision

rule hitとして記録。ONNX lowering未実装のため提出なし。次は既存artifact profile / graph surgeryで削れるか確認する。
