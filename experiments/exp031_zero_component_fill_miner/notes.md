# exp031_zero_component_fill_miner

## Hypothesis

入力の非ゼロセルを壁とみなし、0 connected componentのうち外周に接続する成分を外側色、閉じた成分を内側色で塗れば、region/line fill系の上位taskを説明できる。

## Result

- target tasks: `[198, 187, 203, 313]`
- arc-gen sample: `20`
- pass tasks: `[187]`
- local estimate delta: `0.0`
- submission: no submit, ONNX未生成

## Interpretation

Python ruleとして成立したtaskだけ、次の実験でcost-aware ONNX loweringを検討する。過去のunrolled flood-fillは高costだったため、ONNX化は閉形式のrow/column prefixまたは既存artifact surgeryを優先する。

## Risks

- leakage risk: low-medium: colors are inferred from train only; validation uses test and arc-gen sample20.
- overfitting risk: medium: component-fill rule may still be ARC-family specific and sample20 is not private-like validation.
