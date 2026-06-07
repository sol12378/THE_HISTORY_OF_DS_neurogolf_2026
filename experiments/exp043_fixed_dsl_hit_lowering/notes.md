# exp043_fixed_dsl_hit_lowering

## 仮説

exp042のproxy hitのうち固定shapeのflip/rot/crop/upscale系は、実ONNXへloweringしても既存artifactより低costになり、proxyではなくsubmit候補bundleの改善として採用できる。

## 実験

- base: exp040 + exp041 task145 overlay
- 対象: exp042 hit 9件
- 実装対象: `rot180`, `rot90_ccw`, `crop_tl`, `crop_tr`, `nearest_upscale_3`
- 除外: variable-shape `flip_h/flip_v` と `bbox_nonzero` は次段のobject/shape compilerへ回した
- validation: train + test + all arc-gen

## 結果

- baseline local estimate: 6480.302478
- new local estimate: 6480.302478
- delta: 0.000000
- improved tasks: []
- candidate status counts: {'skipped': 4, 'rejected': 1, 'no_cost_gain': 4}

## 解釈

固定shape DSL loweringが実costで改善できるかを検証した。改善が小さい/ゼロの場合でも、proxyとofficial costの乖離をguardrailとして扱い、次はvariable-shape/object loweringに集中する。

## リスク

- leakage risk: 低。signature lookupは使わず、明示的DSL変換のみ。
- overfitting risk: 低〜中。all arc-gen通過だが、task-specificな固定変換である。

## 次

1. 改善taskを次baseに採用する。
2. variable-shape flip用の幅/高さ検出mask compilerを作る。
3. task031 `bbox_nonzero` の低cost動的bbox loweringを試す。
