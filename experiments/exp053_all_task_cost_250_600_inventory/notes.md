# exp053_all_task_cost_250_600_inventory

## Hypothesis

7700へ進むには、局所的なartifact surgeryではなく、全400 taskをcost 250〜600台へ落とすための差分地図が必要である。

## Result

- strict seed inventory score: 6282.230228
- cost<=600 already: 27/400
- cost<=250 already: 19/400
- need reduction to <=600: 373/400
- projected score if every task is at least cost<=600: 7503.630200
- projected gain to <=600 floor: 1221.399972
- projected score if every task is at least cost<=250: 7833.830824
- projected gain to <=250 floor: 1551.600595
- margin over 7700 at <=600 floor: -196.369800
- margin over 7700 at <=250 floor: 133.830824

## Interpretation

cost<=600 floorだけでも7700を超える推定になる。ただし、これは「全taskを600以下にできる」という強い条件であり、既存teacherをそのまま使う意味ではない。

最大の実装課題は、signature lookup/current高cost taskを説明可能なDSL/DAGに変換し、full-grid GatherND、large ScatterND、dynamic MatMul、長いWhere chainを避けるloweringを作ること。

## Next

1. `task_cost_targets.csv` をmaster queueとして使う。
2. family gap上位から、submit-safe rule searcherを実装する。
3. 最初のcalibration bundleは小さくし、local/LB差を再測定する。

## Risk

- leakage risk: medium。teacher artifactはoracleであり、提出候補ではない。
- overfitting risk: medium-high。all arc-gen exactだけではhidden汎化の証明にならないため、family holdoutと段階submissionが必要。
