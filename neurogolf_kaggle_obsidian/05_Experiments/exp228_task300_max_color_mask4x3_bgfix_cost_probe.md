# exp228_task300_max_color_mask4x3_bgfix_cost_probe

## 目的

exp226の背景channel欠落を修正する。

## 結果

- validation: `0_pass_1_fail`
- padding外側までchannel0=1にしたため失敗。

## 判断

one-hot差分を直接診断する。
