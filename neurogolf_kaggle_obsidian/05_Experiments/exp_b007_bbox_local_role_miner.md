# exp_b007_bbox_local_role_miner

## 目的

P0 sparse fill task (`20, 126, 173, 251, 51, 378, 90, 37, 284`) に対して、bbox-local target coordinate と target color role だけで説明可能ruleを作れるかを確認する。

## 結果

- target task count: `9`
- train-fit candidate: `0`
- evaluated candidate: `0`
- full pass hit: `0`
- local estimate delta: `0.0`
- submission: `no_submit`

## 解釈

固定的なbbox相対座標と単純なcolor roleでは、P0 sparse fillの変更セルを説明できなかった。これは、変更位置が座標だけで決まるのではなく、object/component role、局所frame、対称性、または色役割の条件分岐で決まっている可能性が高いことを示す。

## Risk

- leakage risk: low。train-derived featureだが、arc-gen labelを使っていない。
- overfitting risk: medium。full arc-gen通過前のloweringはしない。

## Decision

full pass candidateがないためONNX loweringなし。次はshape/orbit-conditioned coordinate grammarを試し、それでも駄目ならobject-role predicate minerへ移す。
