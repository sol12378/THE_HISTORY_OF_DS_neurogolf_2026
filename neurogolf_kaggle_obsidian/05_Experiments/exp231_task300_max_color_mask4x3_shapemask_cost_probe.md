# exp231_task300_max_color_mask4x3_shapemask_cost_probe

## 目的

dynamic rectangular shape maskを追加し、可変2x3/3x2/3x3/4x3出力を正しくone-hot padded化する。

## 結果

- validation: `267_pass_0_fail`
- cost: `92595`
- baseline: `77546`

## 判断

正しいがcost超過。中間メモリ削減が必要。
