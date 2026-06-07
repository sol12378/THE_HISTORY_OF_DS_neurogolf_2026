# exp122_focused_surgery_third_pass_limited

## Hypothesis

exp109〜exp111で受理されたtaskだけに絞れば、focused bypass surgeryの三巡目でも探索ノイズを抑えつつ、残った冗長guard/cast/arithmeticを追加で削れる可能性がある。

## Result

- targets: `7`
- generated candidates: `541`
- status counts: `{'rejected': 511, 'improved': 30}`
- accepted tasks: `[184, 187, 263, 316, 394]`
- local delta: `0.014053`
- new local estimate: `6282.950210`

## Decision

三巡目でdeltaが出れば、focused surgeryは提出候補bundleのpost-passとして継続する。deltaが0なら、#25 reviewでこのlaneを補助へ下げ、別familyのfresh compilerへ戻す。

## Risk

- leakage risk: low。
- overfitting risk: low-to-medium。既存artifactの意味保存をfull validationで確認しているが、同一taskへの反復surgeryなのでLB較正が必要。
