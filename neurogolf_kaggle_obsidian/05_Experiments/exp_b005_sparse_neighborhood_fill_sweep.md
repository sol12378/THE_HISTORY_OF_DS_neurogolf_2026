# exp_b005_sparse_neighborhood_fill_sweep

## 目的

P0 sparse fill taskに対して、cost<=250に落としやすいlocal-neighborhood predicateをtrain fitし、全例検証する。

## 結果

- target tasks: `9`
- train-fit candidates: `0`
- full pass hits: `0`

## 判断

raw neighbor-countだけではP0 sparse fillを説明できない。次はbbox-local coordinate、object role、symmetry orbitを組み合わせる。

## Risk

- leakage risk: 低。
- overfitting risk: 中。
