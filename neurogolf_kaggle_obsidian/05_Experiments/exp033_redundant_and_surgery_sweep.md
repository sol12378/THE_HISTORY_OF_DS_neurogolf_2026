# exp033_redundant_and_surgery_sweep

## Hypothesis

exp032で有効だった `And(mask, already_masked)` 型の冗長nodeは、他の高cost artifactにも存在する。
各 `And` nodeの出力を片方の入力に置換してもvalidationが通る場合、1 node分のmemory costを削減できる。

## Result

- base: `experiments/exp032_task187_component_lowering`
- top_k: `120`
- candidate rows: `326`
- improved tasks: `8`
- improved task ids: `80, 86, 145, 187, 255, 268, 376, 398`
- local estimate: `6479.517116`
- delta: `+0.109709`
- gap to 6500: `20.482884`
- submission: no submit, 6500 threshold未達

## Top Improvements

- task376: cost `23413 -> 22513`
- task80: cost `42082 -> 41282`
- task145: cost `69757 -> 68857`
- task255: cost `74403 -> 73503`
- task187: cost `104413 -> 103513`
- task398: cost `84295 -> 83670`

## Interpretation

単純なnode置換でも、sample20 validationとofficial-like scoreを通せば安全に小改善を積み上げられる。
6500までまだ `20.482884` あるため提出しない。
次は `And` 以外の `Or`, `Where`, comparison node、unused initializerの同型探索へ広げる。

## Risks

- leakage risk: medium。graph surgeryはartifactの意味保持をsample20で確認しているが、baseはhigh-risk lookup upper boundを含む。
- overfitting risk: medium。sample20 validationはprivate-like holdoutではない。

## Decision

exp033 bundleを新しいlocal upper boundとして採用する。6500未達のため提出なし。
