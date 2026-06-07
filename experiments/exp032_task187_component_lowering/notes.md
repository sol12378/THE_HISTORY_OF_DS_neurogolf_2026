# exp032_task187_component_lowering

## Hypothesis

task187の既存artifactには、exp031で確認したcomponent-fill規則を実装する過程で冗長なmask nodeが残っている。`safe_name_75` はすでに `safe_name_16` でmask済みなので、`safe_name_76 = And(safe_name_16, safe_name_75)` を `safe_name_75` に置換すればcostを下げられる。

## Result

- baseline cost: `105313`
- candidate cost: `104413`
- validation: `24_pass_0_fail`
- status: `improved`
- local estimate: `6479.407407`
- delta: `0.008583`
- gap to 6500: `20.592593`
- submission decision: `no_submit: below 6500 threshold`

## Interpretation

1 nodeのgraph surgeryでtask187は `105313 -> 104413` に改善した。
6500 thresholdには届かないため提出しない。
この方向は大きな点数にはならないが、既存artifactの小さい冗長削除としてsubmit-safe候補の積み上げに使える。

## Risks

- leakage risk: medium: graph surgery preserves an existing public artifact's semantics; task187 rule was validated separately, but base still includes high-risk lookup artifacts.
- overfitting risk: medium: candidate is validated on train/test/arc-gen sample20, not full private-like holdout.
