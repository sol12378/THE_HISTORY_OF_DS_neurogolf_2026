# exp229_task300_exp228_onehot_diagnostic

## 目的

exp228のmismatchをone-hot tensor差分で診断する。

## 結果

- diff_count: `888`
- max_abs_diff: `1.0`
- out_channel_sums_top4x3: `[4.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- expected_channel_sums_top4x3: `[4.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`

## 判断

result.jsonのfirst_diffsを見て、背景channelまたはpadding/channel順の問題を修正する。
