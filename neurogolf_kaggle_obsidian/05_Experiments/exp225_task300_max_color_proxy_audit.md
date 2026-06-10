# exp225_task300_max_color_proxy_audit

## 目的

task300の最大component crop ruleを、より安い最大nonzero色count + 4x3 mask出力で代替できるか確認する。

## 結果

- max_color_matches_component: `267/267`
- max_color_mask4x3: `267/267`

## 判断

component growthを避けられる。ONNX cost probeへ進む。

## リスク

- leakage risk: low
- overfitting risk: low
