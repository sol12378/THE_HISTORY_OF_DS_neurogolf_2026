# exp_b005_sparse_neighborhood_fill_sweep

## 目的

cost<=250へ近づけるため、P0 sparse fill taskに対してlocal-neighborhood predicateをtrain fitし、全例検証する。

## 結果

- target tasks: 9
- train-fit candidates: 0
- full pass hits: 0
- train/test pass hits: 0
- best partial: None

## 解釈

このgrammarはcost<=250に落としやすいが、現時点でfull passがなければ、近傍数だけでは不足。object role、bbox-local coordinate、symmetry orbitを組み合わせる必要がある。
