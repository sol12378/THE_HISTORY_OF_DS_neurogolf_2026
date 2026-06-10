# exp170_next4m_candidate_validation_audit

## 目的

exp169対象task133/396/158/047について、既存source候補のfull validation / costを事前監査する。

## 結果

- full_ok_total: `8`
- task133: full-ok 3件。exp_b035 distinct候補あり。
- task396: full-ok 1件で実質currentのみ。
- task158: full-ok 2件。exp_b035 distinct候補あり。
- task047: full-ok 2件だが同一raw重複。

## 判断

exp169で特定されたtask133/task158のrepairにはexp_b035 rawを使う。task396/task047は即repairしない。

## リスク

既存public/source candidate由来のためleakage/private robustness riskは高い。
