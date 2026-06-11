# exp329_task185_dynamic_bg_axis_selector_probe

## Hypothesis

exp328 の dynamic bg detector を task185 の dilated axis selector に接続し、検出 bg channel を除外すれば、巨大 row/col template なしで Python selector と一致する。

## Result

- task: `185`
- baseline cost: `59584`
- selector validation: `267_pass_0_fail`
- selector cost: `41519`
- memory / params: `41428` / `91`
- diagnostic ONNX static: `ok`
- scoring ONNX static: `ok`

## Interpretation

selector の correctness は解決した。exp207/208 で見えていた「固定 bg channel のため row/col index がずれる」問題は、dynamic bg detector を selector 前段に入れることで解消された。

cost は baseline 未満で、selector 単体としては task185 full candidate へ進める水準。ただしこれは subgraph probe であり、まだ output core までの full-arc candidate ではない。

## Decision

`no_submit: selector subgraph probe only`

次は selected lattice extraction と homogeneous 2x2 core を接続し、task185 の full candidate を作る。

## Risk

- leakage risk: low。入力のみの modal-color / geometry selector で、output lookup や public feedback は使っていない。
- overfitting risk: medium-low。task-specific rule だが、全 local examples `267/267` で確認済み。

## 5-Experiment Review Note

exp325-329 の短期評価:

- exp325 は fresh public-zero probe C が all-alive で、`franksunp_blended_best` 高 point 順 probe の収率低下を確認した。
- exp326 は task037 が値コピー型で GridSample/shift 候補として妥当なことを確認した。
- exp327 は task037 rule の correctness を証明したが、dense shift-stack cost wall を明確化した。
- exp328 は task185 dynamic bg detector を cost `143` で解いた。
- exp329 は task185 selector を `267_pass_0_fail`、cost `41519` まで進めた。

判断: public-zero は強い新 suspect source が出るまで主軸から下げ、GridSample / solved-rule lowering、特に task185 full candidate を優先する。
