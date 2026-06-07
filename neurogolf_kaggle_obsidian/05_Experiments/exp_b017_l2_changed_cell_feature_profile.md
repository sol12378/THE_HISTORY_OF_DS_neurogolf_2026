# exp_b017_l2_changed_cell_feature_profile

## 目的

L2 top taskのchanged cellをpositive/negative cell datasetとしてprofileし、branch/tree合成で使うべきfeatureを特定する。

## 結果

- target tasks: `5`
- target ids: `5, 133, 173, 285, 286`
- status: `profile_ready`
- submission: `no_submit`

## 主要観察

- task173:
  - best feature: `ray_lr=-1:5`
  - precision `1.0`
  - recall `0.0435`
- task133:
  - best feature: `ray_d1=-4:-3`
  - precision `1.0`
  - recall `0.0225`
- task285:
  - best feature: `ray_d1=-6:1`
  - precision `1.0`
  - recall `0.0123`
- task286:
  - best feature: `ray_lr=-1:6`
  - precision `1.0`
  - recall `0.0109`
- task5:
  - best feature: `ray_lr=2:0`
  - precision `0.6`
  - recall `0.0652`

## 解釈

単一featureで全positiveを拾う形ではないが、ray系featureにはprecision 1.0の小さいpositive islandがある。つまり、L2は「単一predicate」ではなく、複数の高precision branchをORしてpositiveを積み上げるdecision treeが自然。

## Risk

- leakage risk: low。train feature診断のみ。
- overfitting risk: medium。branch/treeは必ずfull arc-gen gateを通す必要がある。

## Decision

次は `high-precision island OR tree miner`。positive-onlyまたは高precision値を貪欲に追加し、train full fitしたtreeだけtest/arc-genへ進める。loweringはray/bbox/neighbor maskの小さいORとして設計する。
