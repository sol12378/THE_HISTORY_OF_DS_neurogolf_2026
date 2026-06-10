# exp241_task271_binary_template_window_audit

## 目的

task271の固定binary signatureを使い、入力内のtemplate一致windowが一意に選べるか監査する。

## 結果

- binary_template: all 3x3 cells nonzero
- target_in_binary_cands: `267/267`
- binary candidate count: always `4`
- best binary rank selector: `closest_tl 73/267`
- color signature templateはほぼ一致せず、target_in_signature_cands `1/267`

## 判断

候補は「4個のfull-nonzero 3x3 block」まで絞れる。次は4候補の色構成selectorを監査する。
