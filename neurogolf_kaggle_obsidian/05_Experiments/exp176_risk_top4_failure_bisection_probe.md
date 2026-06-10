# exp176_risk_top4_failure_bisection_probe

## 目的

subset由来候補が薄くなった後に備え、risk_score上位のtask202/382/205/383をfail-stub化してpublic-zero有無を測る準備をする。

## 結果

- status: `probe_zip_ready`
- expected_drop_if_all_alive: `53.291417344804714`
- expected_lb_if_all_alive: `5940.528582655195`
- zip sanity: pass
- target validation: 4 taskすべて `0_pass_1_fail`

## 判断

exp178後の次probe候補として保持する。

## リスク

risk inventory優先はsubset-sum consensusより較正が弱いため、probe valueは中程度。
