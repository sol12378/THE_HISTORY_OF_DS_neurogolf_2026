# exp019_boundary_flood_fill_campaign

## Hypothesis

背景0を壁で分割するtaskは、境界から到達可能な外側背景と閉領域背景をunrolled ONNXで分ければ、
exp016の高cost public artifactより低costに置換できる。

## Result

- baseline local estimate: `6479.383086`
- new local estimate: `6479.383086`
- delta: `0.000000`
- gap to 6500: `20.616914`
- improved tasks: `[]`

## Risks

- leakage risk: high。baseはexp016でsignature lookupを含む。
- overfitting risk: high。sample-localであり、full arc-gen前の評価。
