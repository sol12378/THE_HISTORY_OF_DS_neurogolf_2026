# exp233_task300_channel_gather_mask4x3_cost_probe

## 目的

selected mask生成を10ch Mulからchannel GatherElementsへ変え、task300 costをbaseline未満へ落とす。

## 結果

- validation: `267_pass_0_fail`
- cost: `52653`
- baseline: `77546`
- local delta: `+0.3871`

## 判断

提出候補。exp234でcurrent best bundleへ組み込む。

## リスク

- leakage risk: low
- overfitting risk: medium-low
