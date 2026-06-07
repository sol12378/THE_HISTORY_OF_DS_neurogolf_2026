# exp_b016_l2_local_predicate_decision_tree_miner

## 目的

L1/L2 sparse fillを、zero cellに対する3x3/5x5近傍・ray・bbox predicateの小さいdecision ruleとして合成する。

## 結果

- target tasks: 5
- predicate count: 37
- candidate predicate sets: 37
- evaluated candidates: 0
- full pass hits: 0
- train/test hits: 0
- train-fit hits: 0

## 判断

full passがあればsmall Conv/mask loweringへ進む。なければ、色roleの決定やnegative samplingを加えたtask-specific treeへ進む。
