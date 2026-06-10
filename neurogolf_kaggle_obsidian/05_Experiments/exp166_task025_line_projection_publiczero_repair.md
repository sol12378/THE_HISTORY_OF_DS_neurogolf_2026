# exp166_task025_line_projection_publiczero_repair

## 目的

exp152 current public bestに、exp165のtask025 guide-line projection candidateを重ねてpublic-zero repair probe zipを作る。

## 結果

- candidate_validation: `266_pass_0_fail`
- candidate_cost: `491600`
- candidate_points_if_public_alive: `11.894579343325983`
- expected_public_lb_if_pass: `5968.174579343326`
- zip sha256: `8468ede68913b54bfd3a75a953ac64c122dd5e4b13017aa4b83447387e720193`
- Kaggle ref: `53523413`
- status: `COMPLETE`
- public LB: `5968.18`

## 判断

exp165はlocal costではno gainだが、task025はpublic-zeroなので正解化すればLB改善が期待できる。full-valid input-only ruleであり、repair probeとして提出し、public LB `5968.18` で成功した。current public bestを更新。

## 成果物

- `experiments/exp166_task025_line_projection_publiczero_repair/result.json`
- `experiments/exp166_task025_line_projection_publiczero_repair/notes.md`
- `experiments/exp166_task025_line_projection_publiczero_repair/submission.zip`

## リスク

- leakage risk: low-to-medium。候補はinput-only ruleだが、task025自体はpublic bisectionで選んだ。
- overfitting risk: medium。完全guide line前提がhidden variantで崩れる可能性がある。
