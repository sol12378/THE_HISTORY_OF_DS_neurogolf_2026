# exp067_public_notebook_utilization_audit

## 目的

ユーザー提示の公開notebook群を、既存blendの再利用ではなく、strict-safe compiler / surgery / task-priorへ変換する。

## 結果

- 既にexp002へ含まれていた系統: `massimilianoghiotto`, `konbu17`, `vyankteshdwivedi`, `magmacot`, `needless090`
- 新規取得: `seddiktrk/surgical-onnx-precision-parameter-reduction`
- 新規取得: `karnakbaevarthur/all-task-description-analysis`
- Karnak dataset: `arc_primitives.csv/json` を取得
- Seddik-style audit scanned: 400 tasks
- unused params total: 31
- duplicate wasted params total: 55
- uniform safe saveable params total: 4583

## 解釈

公開blendはexp002で既にかなり取り込み済み。今後の有意義な使い方は、提出済みartifactをそのまま混ぜることではなく、Seddik式の外科的圧縮をstrict seed post-passへ入れること、Karnakのtask descriptionをcompiler laneのpriorにすること。

## Decision

次は `seddik_surgery_audit.csv` の上位taskをfull-arc gated surgery候補にし、Karnak category/transformationsを `exp054` laneや `exp053` cost queueへjoinする。
