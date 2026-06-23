# exp334_task037_profile

## Hypothesis
Profile task037 changed-cell and cost evidence before designing sparse per-diagonal IR.

## Result
- validation: `266_pass_0_fail`
- changed cell mean: `8.466165413533835`
- baseline cost: `63726`
- cropped dense cost: `721329`

## Decision
No submit. Dense shift-stack is confirmed as a banned lowering pattern for task037.

## Next
Build sparse per-diagonal IR before any ONNX emission.

