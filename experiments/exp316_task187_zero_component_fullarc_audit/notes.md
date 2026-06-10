# exp316_task187_zero_component_fullarc_audit

## Plan

exp031 の task187 zero-component fill rule を全 arc-gen で再検証し、correctness-first ONNX repair へ進む条件を確認する。

## Result

- status: `fullarc_rule_pass`
- border zero component color: `3`
- enclosed zero component color: `2`
- train: `3_pass_0_fail`
- test: `1_pass_0_fail`
- arc-gen: `262_pass_0_fail`

## Act

no_submit_python_rule_audit_only; proceed to correctness-first ONNX lowering if fullarc_rule_pass

## Risk

- leakage risk: low-medium: colors inferred from train only; full arc-gen is used for validation, not fitting.
- overfitting risk: medium: Python rule full-arc pass is necessary but public-zero repair still needs ONNX and LB probe.
