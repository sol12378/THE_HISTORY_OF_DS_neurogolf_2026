# exp230_task300_max_color_mask4x3_padfix_cost_probe

## 目的

padding外側をzeroに戻したtask300候補を再評価する。

## 結果

- validation: `1_pass_1_fail`
- example0は通ったが、可変bbox外側に背景を立ててexample1で失敗。

## 判断

4x3内にdynamic rectangular shape maskが必要。
