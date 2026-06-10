# exp166_task025_line_projection_publiczero_repair

## 目的

exp152 current public bestに、exp165のtask025 guide-line projection candidateを重ねてpublic-zero repair probe zipを作る。

## 結果

- candidate_validation: `266_pass_0_fail`
- candidate_cost: `491600`
- expected_public_lb_if_pass: `5968.174579343326`
- zip sha256: `8468ede68913b54bfd3a75a953ac64c122dd5e4b13017aa4b83447387e720193`
- submission_decision: submit_publiczero_repair_probe
- kaggle_ref: `53523413`
- kaggle_status: `COMPLETE`
- public_lb: `5968.18`

## リスク

low-to-medium: candidate is an input-only geometric rule, but task025 was selected via public-zero bisection.
medium: full-line guide assumption passes all local generated examples but still needs LB probe.
