# exp159_dtype_postpass_candidate_audit

## 目的

exp158でdtype幅が公式costに反映されることを確認したため、current best bundle `exp152` の400 taskから、bool/mask系full-grid中間をFLOATとしてmaterializeしていそうなdtype post-pass候補を構造監査で絞る。

## 結果

- audited_tasks: `400`
- top candidate examples:
  - task366: heuristic_saving `4275000`, boolish_full_grid `129`, where_full_grid `82`
  - task284: heuristic_saving `1899000`, boolish_full_grid `69`
  - task158: heuristic_saving `1494000`, boolish_full_grid `37`, cast_boolish_to_float `18`
  - task382: heuristic_saving `1449000`, boolish_full_grid `51`
  - task187: heuristic_saving `1215000`, boolish_full_grid `45`

## 解釈

多くの高cost artifactはbool/mask系full-grid tensorを大量に持っている。ただし、すでにBOOLとして保持されているものも多く、単純なdtype置換ではなく、FLOATへCastされる箇所やWhere入力周りのmaterializationを局所的に見る必要がある。`task158` は boolish->FLOAT full-grid Cast が18個あり、最初のrewrite監査対象として扱いやすい。

## 判断

次は `task158` または `task366` のCast/consumer patternを詳しく見て、semantic-preservingにFLOAT materializationを減らせるか確認する。候補がなければPhase C-1はgraph rewriteではなく、rule/lowering生成側のdtype設計へ戻す。

## リスク

- leakage risk: low。graph構造監査のみ。
- overfitting risk: low-to-medium。heuristic savingは実rewrite可能性を保証しない。
