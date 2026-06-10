# exp317_task187_zero_component_onnx_repair

## Plan

task187 の public-zero を、exp316 で full-arc pass した zero-component rule の correctness-first ONNX lowering で修復する。

## Result

- status: `repair_zip_ready`
- best_candidate: `{'steps': 30, 'template_name': 'boundary_flood_fill_bg0_30', 'generated_status': 'generated', 'generated_reason': 'bg=0,exterior=3,interior=2,steps=30', 'raw_bytes': 10019, 'static_ok': True, 'static_reason': 'ok', 'validation_status': '266_pass_0_fail', 'validation_reason': 'ok', 'memory': 355500, 'params': 931, 'cost': 356431, 'points_if_public_alive': 12.216104048283515, 'sha256': '2663cf676d4b2ca4656f8f90f0614ceeb980c96d640f859359950013601a8077'}`
- expected_public_lb_if_pass: `6021.176104048283`
- submission_decision: `submit_publiczero_repair_probe`

## Risk

- leakage risk: low-medium: task187 was selected via public-zero probe, but the candidate is an input-derived zero-component rule inferred from train and full-arc validated.
- overfitting risk: medium: correctness-first unrolled flood-fill is task-specific and still needs LB probe to verify public repair.
