# exp137_private_failure_consensus_audit

## 目的

exp136 の subset-sum 候補から、private failure 監査の優先 task queue を作る。

## 結果

- exp136 上位 `20` subset を使用。
- unique candidate task: `56`
- priority queue 上位:
  - task013: top20 freq `20`, exact-gap freq `4`, source `massimilianoghiotto_6254`, sparse/object
  - task002: top20 freq `20`, exact-gap freq `4`, source `massimilianoghiotto_6254`, sparse/object
  - task029: top20 freq `20`, exact-gap freq `4`, source `massimilianoghiotto_6254`, crop/resize
  - task009: top20 freq `20`, exact-gap freq `4`, source `massimilianoghiotto_6254`, sparse/object
  - task024: top20 freq `19`, exact-gap freq `4`, source `massimilianoghiotto_6254`, sparse/object
  - task018: top20 freq `18`, exact-gap freq `4`, source `kojimar_5800_minimal_blend`, sparse/object

## 解釈

上位 consensus は task013/002/029/009 に集中した。subset-sum の偏りはあるが、これらは高risk score・高cost・shape variability を持つため、private failure監査の初手として妥当。次は task013/002/029/009 の代替source探索、既存arc-gen全例の失敗境界、shape外挿リスクを個別に見る。

## 成果物

- `experiments/exp137_private_failure_consensus_audit/consensus_audit.csv`
- `experiments/exp137_private_failure_consensus_audit/result.json`

## リスク

- leakage risk: low。
- overfitting risk: medium。consensus は subset-sum 由来であり、private failure の直接証明ではない。
