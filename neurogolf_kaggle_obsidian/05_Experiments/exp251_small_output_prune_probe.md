# exp251_small_output_prune_probe

## 目的

small-output solved/near-solved候補 task100/242/253/271 の既存artifactに未使用initializerが残っていれば削除し、score改善するか確認する。

## 結果

- targets: `100, 242, 253, 271`
- removed_initializers: all `0`
- validation: all pass
- candidate_cost: all baselineと同じ
- improved_tasks: `[]`

## 判断

既存artifactに単純な未使用initializerはない。より深いsurgeryまたは新規候補探索へ戻る。

## リスク

semantics-preserving cleanupなのでleakage/overfitting riskは低いが、効果もなかった。
