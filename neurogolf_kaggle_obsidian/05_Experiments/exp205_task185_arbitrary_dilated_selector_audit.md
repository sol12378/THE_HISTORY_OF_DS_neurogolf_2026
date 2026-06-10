# exp205_task185_arbitrary_dilated_selector_audit

## 目的

exp204のmismatchが、detected grid-line windowsではなく任意のdilated startをscoreしていることに由来するか確認する。

## 結果

- valid_selector_pass: `267/267`
- arbitrary_selector_pass: `267/267`
- arbitrary_match_valid: `267/267`

## 判断

任意dilated start selector自体はPython基準では正しい。exp204 mismatchは、ONNX上のscore source、index mapping、core/output表現のどこかにある。

## リスク

- leakage risk: low。
- overfitting risk: medium-low。
