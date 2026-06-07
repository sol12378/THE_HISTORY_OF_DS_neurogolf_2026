# exp_b012_task037_teacher_delta_calibration

## 目的

strict seedにtask037 teacher artifactだけを差し替え、local/LB差を単一taskで測る。これは説明可能rule採用ではなく、LB calibration用の高リスクdelta提出である。

## 結果

- strict task cost: 63726
- teacher task cost: 8379
- local estimate delta: 2.028864
- new local estimate: 6284.259092
- validation: 24_pass_1_fail, ok=False, reason=mismatch example 24

## Risk

- leakage risk: high。teacher artifactはfinal戦略ではない。
- overfitting risk: high。Kaggle LB deltaでのみ較正可能。

## Decision

validationが通れば、single-task delta calibrationとして提出する。LBで崩れる場合、task037 teacher系はhidden非対応とみなす。
