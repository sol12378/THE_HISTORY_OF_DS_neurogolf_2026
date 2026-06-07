# exp_b015_l1_l2_sparse_fill_rule_bank_sweep

## 目的

exp_b010でtask037 full passを出したcomponent/object-role sparse fill rule bankを、signature lookup taxonomyのL1/L2 laneへ横展開する。

## 結果

- fast top k: `20`
- target tasks: `20`
- lane counts:
  - `L2_local_predicate_sparse_fill`: `18`
  - `L1_static_sparse_background_fill`: `2`
- rule count: `13`
- evaluated candidates: `260`
- full pass hit: `0`
- train/test hit: `0`
- train-fit hit: `0`
- candidate status: all `partial`
- submission: `no_submit`

## 解釈

b010のcomponent/object-role単一ruleはtask037には刺さったが、gain上位L1/L2には横展開しなかった。上位L2は、単純なrow/column span、ray、bbox、hole componentではなく、3x3/5x5 local predicate、color-role condition、またはsmall decision treeが必要。

## Risk

- leakage risk: low-to-medium。説明可能ruleを評価したのみ。
- overfitting risk: medium。arc-genは評価のみだが、次のdecision tree合成ではholdout/全arc-gen gateが必要。

## Decision

次は `L2 local predicate decision tree miner`。changed cellをpositive、background unchangedをnegativeとして、3x3/5x5近傍・色役割・component roleから小さいpredicate treeを合成する。full pass後のみsmall Conv/mask loweringへ進む。
