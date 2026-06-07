# exp_b016_l2_local_predicate_decision_tree_miner

## 目的

L2 sparse fillを、zero cellに対する3x3/5x5近傍・ray・bbox predicateの小さいdecision ruleとして合成する。

## 結果

- fast top k: `5`
- target tasks: `5`
- predicate count: `37`
- predicate set count: `37`
- screened candidates: `185`
- train exact-fit: `0`
- evaluated full candidates: `0`
- full pass hit: `0`
- submission: `no_submit`

## 解釈

gain上位L2 taskは、単一の3x3/5x5近傍predicateやbbox/ray predicateではtrain例にも一致しなかった。b015よりさらに細かく見ても、単一局所predicate路線は弱い。

## Risk

- leakage risk: low-to-medium。train screenのみ。
- overfitting risk: medium。次のdecision tree合成では必ずfull arc-gen gateが必要。

## Decision

次は「changed cell分類」そのものをデータとして扱い、positive/negative cellのfeature importanceを読む。target color role、変更セルのrow/col分布、近傍色の組み合わせ、component idを分析してから、branch付きdecision treeを合成する。
