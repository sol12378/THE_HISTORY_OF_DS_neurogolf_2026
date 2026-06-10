# exp165_task025_line_projection_onnx_probe

## 目的

exp164で見つかったtask025 guide-line projection ruleをcorrectness-first ONNXに落とし、full validationとofficial costを測る。

## 結果

- status: `no_cost_gain`
- validation_status: `266_pass_0_fail`
- baseline_cost: `89286`
- candidate_cost: `491600`
- decision: do_not_adopt_refine_lowering

## リスク

low: input-only rule lowered to ONNX; no labels or raw data beyond provided validation examples are embedded.
medium: rule assumes complete guide lines and may need hidden robustness review before final-safe classification.
