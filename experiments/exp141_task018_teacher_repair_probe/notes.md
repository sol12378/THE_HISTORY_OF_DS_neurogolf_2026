# exp141_task018_teacher_repair_probe

## Hypothesis

exp140でtask018がpublic-zeroと推定された。exp023 teacher artifactへtask018だけ差し替えれば、public LBで約+15.23を回収できる可能性がある。

## Result

- status: `repair_probe_blocked`
- teacher validation: `24_pass_1_fail`
- teacher cost: `17467`
- expected_lb_if_public_pass: `5945.781931334572`
- zip sha256: `a09db509c3bfa6d3ea0ad0cec402d549a0719b0dd92ae7e7a37e2deec0e0b6f9`

## Decision

do_not_submit

## Leakage / Overfitting Risk

high: exp023 teacher artifact is signature-lookup-derived and previously part of a collapsed high-risk bundle.
high: public pass would not prove private robustness; use only as single-task calibration/repair evidence.
