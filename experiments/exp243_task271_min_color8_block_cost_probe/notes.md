# exp243_task271_min_color8_block_cost_probe

## 目的

exp242で見つけたtask271 rule「3x3 full-nonzero block 4候補からcolor8_count最小を選ぶ」をONNX化し、baseline cost `28991` より安くなるか確認する。

## 結果

- validation: `267_pass_0_fail`
- status: `no_cost_gain`
- candidate_cost: `70847`
- baseline_cost: `28991`
- reason: `candidate cost is not lower than baseline`

## 判断

improvedならsingle-task delta提出候補。no_cost_gainならsolved assetとして保持し、別候補へpivotする。
