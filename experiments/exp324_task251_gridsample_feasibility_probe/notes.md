# exp324_task251_gridsample_feasibility_probe

## 目的

task251 GridSample queue の最初の score-direct probe として、実ONNX `GridSample` が NeuroGolf official scorer で使えるか、また純粋な data movement だけで task251 を解ける余地があるか確認する。

## 結果

- Python rule: `266_pass_0_fail`
- base cost(rescored): `100580`
- GridSample variants:
  - opset16: `{'name': 'identity_gridsample_opset16', 'opset': 16, 'file_bytes': 7533, 'sha256': '1bfc640246a169a9a1cf009f4908528c4a2701802607b0baababb1ff401809ca', 'static_ok': True, 'static_reason': 'ok', 'sanitize_ok': True, 'sanitized_file_bytes': 7537, 'sanitized_sha256': '198b19e290e94d094e67be0e160d72d8c8009609a9d4855d80d614405e9aa833', 'validation_ok': False, 'validation_status': '0_pass_1_fail', 'validation_reason': 'mismatch example 0', 'score_reason': 'ok', 'memory': 0, 'params': 1800, 'cost': 1800, 'points': 17.504458056115745, 'delta_if_correct_vs_base': 4.023166765841701, 'expected_lb_if_correct': 6012.983166765842}`
  - opset20: `{'name': 'identity_gridsample_opset20', 'opset': 20, 'file_bytes': 7533, 'sha256': 'f9693a354f3cc0ebe73a7c5a3f1777f585acef7fe77ce49167f5352023ee8321', 'static_ok': True, 'static_reason': 'ok', 'sanitize_ok': True, 'sanitized_file_bytes': 7537, 'sanitized_sha256': 'd6ba319cf2e4f0f8c94ea278b5f5bf0ce462c026b7349b21886543e30263231c', 'validation_ok': False, 'validation_status': '0_pass_1_fail', 'validation_reason': 'mismatch example 0', 'score_reason': 'ok', 'memory': 0, 'params': 1800, 'cost': 1800, 'points': 17.504458056115745, 'delta_if_correct_vs_base': 4.023166765841701, 'expected_lb_if_correct': 6012.983166765842}`
- input lacks color1 examples: `266` / `266`

## 判断

GridSample is official-scoreable with best observed identity cost 1800, but pure sampling is insufficient for task251 because many inputs lack color1 while outputs create color1 fills. Next candidate must generate a closed-component mask and combine it with a cheap color1/recolor primitive, or pivot to another GridSample target.

## Submission

`no_submit: feasibility probe only; identity GridSample is intentionally not task-correct.`

## Risk

- leakage risk: low: uses only official task examples for rule revalidation and an identity primitive probe; no hidden/public feedback.
- overfitting risk: low-to-medium: task251 rule is task-specific but already full-arc validated; this experiment does not submit a replacement.
