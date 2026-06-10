# exp205_task185_arbitrary_dilated_selector_audit

## 目的

exp204のmismatchが、detected grid-line windowsではなく任意のdilated startをscoreしていることに由来するか確認する。

## 結果

- valid_selector_pass: `267/267`
- arbitrary_selector_pass: `267/267`
- arbitrary_match_valid: `267/267`

## 判断

arbitrary selectorが低ければ、ONNX candidateにはgrid-line window maskまたはcompact valid-window templateが必要。

## リスク

- leakage risk: low。
- overfitting risk: medium-low。
