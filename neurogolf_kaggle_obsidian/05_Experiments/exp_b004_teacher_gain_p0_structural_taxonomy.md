# exp_b004_teacher_gain_p0_structural_taxonomy

## 目的

teacher-gain P0 18 taskの構造を分類し、cost<=250へ向けて次に追加すべきgrammarを決める。

## 結果

Grammar counts:

- sparse background fill with object-role/orbit grammar: `1`
- sparse color-role fill with local neighborhood predicates: `8`
- shape-transform/object crop grammar: `7`
- object movement/copy grammar: `1`
- general sparse/object completion grammar: `1`

## 判断

P0は単純幾何ではなく、bbox-local coordinate、object role、symmetry、shape ruleを組み合わせる必要がある。cost<=250に近い候補は、tiny ScatterND / small static mask / small Convで表せるsparse fill系。

## Risk

- leakage risk: 低。構造診断のみ。
- overfitting risk: 中。all arc-genを診断に使用。
