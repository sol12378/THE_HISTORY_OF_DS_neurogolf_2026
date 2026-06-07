# exp_b007_bbox_local_role_miner

## 目的

cost<=250を意識し、P0 sparse background fill taskに対してbbox-local coordinateとtarget color roleを組み合わせた説明可能ruleを探索する。

## 結果

- target tasks: 9
- train-fit candidates: 0
- full pass hits: 0
- train/test pass hits: 0
- best partial: None

## 判断

full pass hitがあればtiny coordinate loweringへ進む。なければ、static relative positionsだけでは不足なので、shape-conditioned / orbit-conditioned relative positionsを追加する。
