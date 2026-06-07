# exp_b009_bbox_affine_formula_miner

## 目的

P0 sparse fill taskのtarget cellを、bbox height/widthからの affine-like coordinate formula (`0`, `1`, `n-1`, `n`, `2n±1`, `mid±1` など) とtarget color roleで説明できるかを検証する。

## 結果

- target task count: `9`
- evaluated candidate: `0`
- full pass hit: `0`
- train/test pass hit: `0`
- best partial: none
- local estimate delta: `0.0`
- submission: `no_submit`

## 解釈

bbox affine coordinate grammarは候補すら生成できなかった。b007の固定座標、b008のshape table、b009のaffine式がすべて不発なので、P0 sparse fillはcoordinate-first探索ではなく、component/object/color-role selectionを先に解くべき。

## Risk

- leakage risk: low-to-medium。formulaは説明可能だがtrain-derived。
- overfitting risk: medium。全arc-gen full pass前に採用しない。

## Decision

次PDCAは `component/object-role sparse fill miner`。候補は、nonzero component、hole/border, endpoints, symmetry orbit, nearest color object, changed-color role を組み合わせ、full passしたものだけtiny coordinate/mask ONNXへloweringする。
