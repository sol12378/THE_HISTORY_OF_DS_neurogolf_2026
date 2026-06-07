# exp_b009_bbox_affine_formula_miner

## 目的

bbox height/widthから変更セル座標を生成するaffine-like formulaを探索する。shape-conditioned lookupより説明可能で、cost<=250 loweringに近い。

## 結果

- target tasks: 9
- evaluated candidates: 0
- full pass hits: 0
- train/test pass hits: 0
- best partial: None

## 判断

full passがあればaffine coordinate generatorとしてloweringへ進む。なければ、bbox affineだけでは不足で、object-role target selectionへ移る。
