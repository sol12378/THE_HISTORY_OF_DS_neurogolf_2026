# exp151_task023_candidate_validation_audit

## 目的

exp150でpublic-zeroと推定されたtask023について、既存source候補で修復できるかを監査する。

## 結果

- source rows: `33`
- unique raw candidates: `5`
- full_ok candidates: `2`
- current raw: `28e232e6ced5220f2426f9f805c049e0556f69e037dd2dbcfe8efa62c735edb5`
- current raw source: `franksunp_blended_best`
- alternate full_ok raw: `exp_b035_new_source_full_arc_blend`
- alternate raw sha256: `381d1261e875c1cddec3fdc4fdca4eff331d3edad2677ad6ba7a97a0d16be814`
- alternate official cost: `72612`
- alternate validation: `266_pass_0_fail`

## 解釈

現行rawはexp150でpublic-zeroと推定済み。`exp_b035` に別rawのfull validation候補があり、task018と同様のsource replacement repairが可能。

## 判断

`exp_b035` task023 rawを使い、exp144 current public bestに重ねる単独repair probeを作る。

## リスク

- leakage risk: high。既存public/source blend由来のため、private-safeとは扱わない。
- overfitting risk: high。public bisectionで特定したtaskのpublic repairであり、最終採用にはリスクレビューが必要。
