# exp175_next2o_candidate_validation_audit

## 目的

exp174対象task285/286について、既存source候補のfull validation / costを事前監査する。

## 結果

- full_ok_total: `4`
- task285: full-ok 2件。exp_b035 distinct候補あり。ただしcost `395468` / points `12.112175`。
- task286: full-ok 2件だがcurrent系の微小差分で、public-zero repair候補としては弱い。

## 判断

exp174でtask285 public-zeroが出たため、exp_b035 rawでrepair probeを作成する。

## リスク

既存public/source candidate由来のためleakage/private robustness riskは高い。
