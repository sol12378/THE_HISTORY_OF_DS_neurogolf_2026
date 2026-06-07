# exp057_l1_object_role_feature_miner

## Hypothesis

task020はbbox/object-role boolean featureのAND ruleで説明できる。

## Result

- target: task020
- candidate feature sets: `10`
- saved evaluated candidates: `0`
- full pass hits: `0`
- train/test pass hits: `0`

## Interpretation

AND feature ruleはtrainでも十分に通らず、表現が硬すぎる。次は正解変更templateを分類し、少数template selectorとしてruleを組み直す。

## Decision

ONNX loweringなし。提出なし。
