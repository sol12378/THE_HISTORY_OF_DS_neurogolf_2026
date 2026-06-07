# exp109_task048_artifact_surgery_sweep

## Hypothesis

task048はbridge connectivity ruleで解けているが、naive 8x8 connectivity loweringはbaselineより高cost。既存artifactには既に必要な構造があり、generated guard/castの局所surgeryでsafe micro-deltaを拾える可能性がある。

## Result

- campaign #: `11`
- task: `48`
- baseline cost: `5609`
- best candidate: `bypass_Cast_006_in0`
- candidate cost: `5481`
- validation: `270_pass_0_fail`
- local delta: `+0.023084884462`
- new local estimate: `6282.835302884462`
- accepted tasks: `[48]`
- submission decision: `submit_candidate_after_review`

## Interpretation

#10後の方針転換が初めてlocal estimateを動かした。広いanchor scanではなく、rule-hit済みtaskの既存artifact focused surgeryに寄せる方が、少なくともsafe micro-deltaにはつながる。

gainは小さいため7500には遠いが、LB較正候補として価値がある。次は同じfocused surgeryをtask048周辺または他のrule-hit taskへ広げるか、task048のaccepted candidateをcurrent safe bundleへunionしてsubmit判断する。

## Risk

- leakage risk: low。graph surgeryのみ。
- overfitting risk: low-to-medium。full validation済みだが、single-task LB calibrationで確認する価値がある。
