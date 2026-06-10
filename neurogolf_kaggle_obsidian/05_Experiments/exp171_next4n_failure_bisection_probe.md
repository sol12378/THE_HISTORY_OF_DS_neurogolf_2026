# exp171_next4n_failure_bisection_probe

## 目的

exp169が全aliveだった場合に備え、次候補task003/038/001/086のfail-stub probe zipを事前準備する。

## 結果

- status: `probe_zip_ready`
- expected_drop_if_all_alive: `65.07777305503902`
- expected_lb_if_all_alive: `5903.102226944961`
- zip sanity: pass
- target validation: 4 taskすべて `0_pass_1_fail`

## 判断

exp172が先にscore-producing repairとなったため提出保留。次のbisection slot候補として保持。

## リスク

提出する場合はsubmission枠を消費する。public behavior測定なのでoverfitting riskは中。
