# exp173_next4n_candidate_validation_audit

## 目的

exp171対象task003/038/001/086について、既存source候補のfull validation / costを事前監査する。

## 結果

- full_ok_total: `8`
- task003: full-ok 1件。明確なdistinct repair sourceは少ない。
- task038: full-ok 1件。明確なdistinct repair sourceは少ない。
- task001: full-ok 2件。beicicc候補あり。
- task086: full-ok 4件。accepted surgery variantsあり。

## 判断

exp171を提出してpublic-zeroが出た場合、task001/task086は既存候補で即repair可能性あり。task003/task038はrule/source再探索が必要。

## リスク

既存public/source candidate由来のためleakage/private robustness riskは中〜高。
