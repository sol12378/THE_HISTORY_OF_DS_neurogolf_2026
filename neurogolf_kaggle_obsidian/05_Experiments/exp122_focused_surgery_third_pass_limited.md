# exp122_focused_surgery_third_pass_limited

## Purpose

exp109〜exp111で受理された7 taskだけに限定し、focused bypass surgeryの三巡目がまだsubmit-safe localを改善するか確認する。

## Result

- targets: 7
- generated candidates: 541
- accepted tasks: `[184, 187, 263, 316, 394]`
- local delta: `+0.014053`
- new local estimate: `6282.950210`
- submission.zip: generated

## Interpretation

focused surgeryは三巡目でも微小に効く。特にtask187は `103513 -> 102613` とまだ900 cost削れた。ただし、同じtask群への反復でありgainは小さいため、LB 1位を狙う主戦力ではなく、提出前post-pass / LB calibration laneとして扱う。

## Risk

- leakage risk: low。既存graph surgeryのみでfull validation gated。
- overfitting risk: low-to-medium。同一task反復なのでKaggle micro calibrationが必要。

## Decision

exp122 bundleは提出候補として保持する。ただし#25 reviewでは、主方針を低cost primitive compilerへ戻す。
