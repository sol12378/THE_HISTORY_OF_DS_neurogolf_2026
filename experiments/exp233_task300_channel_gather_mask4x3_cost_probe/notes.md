# exp233_task300_channel_gather_mask4x3_cost_probe

## 目的

task300 max-color proxyのselected mask生成を10ch Mulからchannel GatherElementsへ変更し、exp232 cost `81541` をbaseline `77546` 未満へ下げられるか確認する。

## 結果

- validation: `267_pass_0_fail`
- status: `improved`
- candidate_cost: `52653`
- baseline_cost: `77546`
- reason: `ok`

## 判断

improvedならbundle/submission候補。no_cost_gainならtask300 max-color laneは一旦保留。
