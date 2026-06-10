# exp138_priority_task_alternative_source_audit

## 目的

exp137 の private failure consensus 上位 task `013/002/029/009/024/018` について、既存 manifest と candidate_eval から代替sourceや修復候補があるかを棚卸しする。

## 結果

- targets: `13, 2, 29, 9, 24, 18`
- manifest rows: `203`
- candidate_eval rows: `199`
- 6 taskすべてで複数sourceが見つかった。
- ただし best alternative は `exp004_public_blend_relaxed_static` で、cost は現行と同値。直接修復候補というより、同一/近縁 artifact の再掲である可能性が高い。
- candidate_eval には full-pass no-gain が各taskに少数あるが、改善candidateはなかった。

## 解釈

既存実験内の単純な代替source差し替えだけでは、top consensus task の修復はまだ見えない。private failure 仮説を直接検証するには、bisection probe の情報価値が高い。

## 判断

次は `task013/002/029/009` を fail stub に差し替える group probe を作成し、LB drop からこれらが現public scoringで生きているかを測る。

## リスク

- leakage risk: low。既存manifest監査のみ。
- overfitting risk: medium。代替sourceは同じhidden failure modeを共有する可能性が高い。
