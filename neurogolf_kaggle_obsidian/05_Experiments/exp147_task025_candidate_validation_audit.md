# exp147_task025_candidate_validation_audit

## 目的

exp146でpublic-zeroと推定されたtask025について、既存source候補で修復できるかを監査する。

## 結果

- source rows: `34`
- unique raw candidates: `5`
- full_ok candidates: `1`
- full_ok は `franksunp_blended_best` / current raw のみ。
- official cost: `89286`
- validation: `266_pass_0_fail`

## 解釈

task025の既存full-ok候補は現行public-zero候補だけだった。task018のような別raw差し替えrepairは現時点で見つからない。

## 判断

task025は既存source差し替えではなく、rule repair / robust lowering が必要。短期LB最大化では、次のpublic-zero task探索を継続する。

## リスク

- leakage risk: medium-to-high。
- overfitting risk: medium-to-high。
