# exp_b008_shape_conditioned_sparse_fill

## 目的

P0 sparse fill taskに対して、bbox shape / color signature ごとの小さいbranch tableで説明可能ruleを作れるかを検証する。

## 結果

- target task count: `9`
- evaluated candidate: `21`
- full pass hit: `0`
- train/test pass hit: `0`
- best partial: task126 `bbox_shape:const_4`
  - train pass: `3/3`
  - test pass: `0/1`
  - arc pass: `11/262`
  - total pass: `14/266`
  - fail reasons: `unseen_key`, `cell_oob`
- local estimate delta: `0.0`
- submission: `no_submit`

## 解釈

shape-conditioned branch tableはtrainには合わせられるが、arc-genで未知shapeが大量に出る。これは「小さいlookupとしてのdecision tree」はまだmemorization寄りで、hidden生成に耐えるruleではないことを示す。

## Risk

- leakage risk: medium。branchはtrain-derivedで、tableが大きくなるとlookup化する。
- overfitting risk: medium-high。全arc-gen full passとKaggle delta確認なしでは採用不可。

## Decision

shape lookupだけでは進めない。次はbbox height/width affine formulaを試し、それでも駄目ならgenerative object-role grammarへ移行する。
