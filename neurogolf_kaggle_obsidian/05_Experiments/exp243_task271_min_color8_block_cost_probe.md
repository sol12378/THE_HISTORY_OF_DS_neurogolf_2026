# exp243_task271_min_color8_block_cost_probe

## 目的

exp242のtask271 ruleをONNX化し、baseline cost `28991` より安くなるか確認する。

## 結果

- validation: `267_pass_0_fail`
- candidate_cost: `70847`
- baseline_cost: `28991`
- status: `no_cost_gain`
- 実装: 3x3 Convでfull-nonzero window/color8_countを計算し、ArgMinでoffset選択、GatherElementsでcrop。

## 判断

ruleは正しいが、現行 Conv + dynamic GatherElements lowering は高すぎる。task271はsolved-rule assetとして保持し、提出しない。

## リスク

cost wallはlowering由来。より安い4-block専用selectorが作れれば再開余地あり。
