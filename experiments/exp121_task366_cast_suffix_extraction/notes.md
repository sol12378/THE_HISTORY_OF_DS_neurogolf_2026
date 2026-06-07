# exp121_task366_cast_suffix_extraction

## Hypothesis

task366のfinal-shape mask/type intermediateは、`Cast -> output` だけを足せば正しいfloat outputとして使える可能性がある。

## Result

- candidates: `6`
- status counts: `{'rejected': 6}`
- validation counts: `{'0_pass_1_fail': 6}`
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.936157`

## Interpretation

Cast suffixで通れば、task366既存artifactの大きなsuffixを削れる。通らなければ、mask intermediateは最終outputと意味的に違うため、このlaneはpivotする。

## Risk

- leakage risk: low。
- overfitting risk: low-to-medium。
