# exp177_risk_top4_candidate_validation_audit

## 目的

exp176対象task202/382/205/383について、既存source候補のfull validation / costを事前監査する。

## 結果

- full_ok_total: `8`
- task202/382: full-okはあるが同一raw寄り。
- task205: current-only。
- task383: distinct候補が複数あり、repair surfaceが比較的良い。

## 判断

exp176でpublic-zeroが出た場合、task383なら既存source repairを優先検討。202/382/205はrule/source再探索寄り。

## リスク

既存public/source candidate由来のためleakage/private robustness riskは中〜高。
