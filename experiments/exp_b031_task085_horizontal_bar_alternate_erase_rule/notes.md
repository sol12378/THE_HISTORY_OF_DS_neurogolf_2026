# exp_b031_task085_horizontal_bar_alternate_erase_rule

## 目的

`exp071` で object_erase_or_mask_clean と判定された task085 を説明可能ruleへ圧縮する。

## 結果

- validation: `265/265`
- train: `2/2`
- test: `1/1`
- arc-gen: `262/262`
- avg changed cells: `19.5585`

## Rule

同一色で構成されたheight 3のsolid horizontal rectangleを見つけ、その中央行だけ、component左端から奇数offsetのセルを0へ消す。

## Lowering Note

global checkerboard parityではなく、component left edgeからのrelative parityが必要。次は既存artifact profileかrun-left parity loweringを試す。
