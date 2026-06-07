# exp115_task185_lattice_position_inventory

## Hypothesis

task185の固定lattice loweringが失敗した理由は、4x4 lattice位置が例ごとに変わるためである。dynamic lowering前に、位置pattern数とshape依存性を測る。

## Result

- examples: `267`
- unique position patterns: `46`
- unique delta patterns: `3`
- shape counts: `{'27x27': 83, '29x29': 184}`
- position patterns by shape: `{'27x27': 9, '29x29': 37}`

## Interpretation

position patternが少なければshape-branch static Sliceで進める。多ければ、grid-line detectionをONNXで計算する必要がある。

## Risk

- leakage risk: low。
- overfitting risk: medium。raw position tableをそのまま提出候補にしない。
