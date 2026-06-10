# exp231_task300_max_color_mask4x3_shapemask_cost_probe

## 目的

task300 max-color proxyにdynamic rectangular shape maskを加え、可変2x3/3x2/3x3/4x3 outputをone-hot padded形式で正しく出せるか確認する。

## 結果

- validation: `267_pass_0_fail`
- status: `no_cost_gain`
- candidate_cost: `92595`
- baseline_cost: `77546`
- reason: `candidate cost is not lower than baseline`

## 判断

improvedならbundle/submission候補。no_cost_gainなら正しいruleでもこのloweringはcost wall。
