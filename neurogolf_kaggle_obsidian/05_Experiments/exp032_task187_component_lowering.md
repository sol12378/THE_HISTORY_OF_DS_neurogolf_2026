# exp032_task187_component_lowering

## Hypothesis

task187の既存artifactには、exp031で確認したcomponent-fill規則を実装する過程で冗長なmask nodeが残っている。
`safe_name_75` はすでに `safe_name_16` でmask済みなので、`safe_name_76 = And(safe_name_16, safe_name_75)` を `safe_name_75` に置換すればcostを下げられる。

## Result

- base: `experiments/exp023_graph_surgery_exp016`
- task: `187`
- baseline cost: `105313`
- candidate cost: `104413`
- validation: `24_pass_0_fail`
- local estimate: `6479.407407`
- delta: `+0.008583`
- gap to 6500: `20.592593`
- submission: no submit, 6500 threshold未達

## Interpretation

1 nodeのgraph surgeryでtask187を改善できた。
大きな点数ではないが、既存artifactの冗長mask削除が有効であることを確認した。
次は同型の冗長 `And(mask, already_masked)` を他の高cost artifactへ横展開する。

## Risks

- leakage risk: medium。既存public artifactの意味を保つgraph surgeryで、baseはhigh-risk lookup upper boundを含む。
- overfitting risk: medium。train/test/arc-gen sample20で検証したがprivate-like holdoutではない。

## Decision

6500未達のため提出しない。次PDCAは高cost artifact全体の冗長mask graph surgeryを探索する。
