# exp_b003_teacher_gain_p0_rule_sweep

## 目的

exp048のteacher-gain P0 18 taskを対象に、同じ説明可能rule familyを再評価する。

## 結果

- scanned tasks: `18`
- evaluated rules: `54`
- full pass hits: `0`
- train/test pass hits: `0`
- best partial: task020 `square_d4_orbit_completion`, `132/266`

## 判断

対象設定は正しくなったが、D4/rectangle/lineだけでは不十分。P0 sparse/object tasksにはobject-role grammarが必要。

## Risk

- leakage risk: 低。
- overfitting risk: 中。
