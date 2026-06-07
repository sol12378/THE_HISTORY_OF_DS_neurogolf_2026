# exp115_task185_lattice_position_inventory

## Hypothesis

task185の固定lattice loweringが失敗した理由は、4x4 lattice位置が例ごとに変わるためである。dynamic lowering前に、位置pattern数とshape依存性を測る。

## Result

- campaign index: `17`
- examples: `267`
- input shapes: `29x29=184`, `27x27=83`
- unique raw position patterns: `46`
- position patterns by shape: `27x27=9`, `29x29=37`
- unique delta/spacing patterns: `3`

Spacing counts:

- `3,3,3`: `96`
- `5,5,5`: `88`
- `4,4,4`: `83`

## Interpretation

raw `(rr, cc)` tableは46 patternあり、そのままbranch tableにするとlookup-likeで危険。ただしspacingは3種類しかない。したがって、task185 compilerはraw position tableではなく、grid-line spacing/offset detectionとして設計する。

27x27はspacing `4`、29x29はspacing `3` または `5` が中心に見える。次はshape/line statisticsからspacingとoffsetを低costに選ぶprobeを作る。

## Next Action

#18では、spacing 3/4/5 の候補それぞれについて低cost static Slice/Conv/Padを作り、input shapeまたはgrid-line countからbranch selectできるかを測る。

## Risk

- leakage risk: low。input geometry inventoryのみ。
- overfitting risk: medium。raw position tableを提出候補にしない。
