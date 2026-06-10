# exp186_task187_massimiliano_repair_probe

## 結果

- Kaggle ref: `53526998`
- public LB: `6005.90`
- expected if repair works: `6019.37`
- base exp178 public LB: `6005.93`
- validation: `266_pass_0_fail`

## 判断

期待gainは出ず、exp178を下回ったため採用しない。task187は未解決public-zero repair queueに戻す。

## リスク

full-local-validでもpublic repairになるとは限らないことを確認した。
