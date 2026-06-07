# exp_b012_task037_teacher_delta_calibration

## 目的

strict seedにtask037 teacher artifactだけを差し替え、local/LB差を単一taskで測れるか確認する。これは説明可能rule採用ではなく、LB calibration用の高リスクdelta提出候補である。

## 結果

- strict task cost: `63726`
- teacher task cost: `8379`
- local estimate delta if accepted: `+2.028864`
- validation: `24_pass_1_fail`
- status: `rejected_validation_failed`
- submission: `no_submit`

## 解釈

task037 teacherはinventory上は大きく良いが、全arc-genで不一致が出る。したがってsingle-task LB calibrationにも使わない。

## Risk

- leakage risk: high。teacher artifactはfinal戦略ではない。
- overfitting risk: high。今回はlocal full validationで落ちたため提出不可。

## Decision

teacher delta提出なし。task037はb010 ruleを低cost loweringする方向に戻す。
