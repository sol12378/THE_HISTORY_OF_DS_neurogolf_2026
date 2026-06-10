# exp246_task100_mode_failure_audit

## 目的

task100のmode_nz rule `248/266` の失敗18例を監査し、count rankやbbox特徴で切替条件が作れるか確認する。

## 結果

- mode_low: `248/266`
- target_rank_hist: `{0: 248, 1: 18}`
- best rule: `bbox_area_max 266/266`

## 判断

task100 ruleを発見。非zero色のうちbbox areaが最大の色を選び、2x2全同色templateを出力する。

## リスク

bbox area tie-breakはhiddenでedgeがありうるが、入力構造ruleでありlookupではない。
