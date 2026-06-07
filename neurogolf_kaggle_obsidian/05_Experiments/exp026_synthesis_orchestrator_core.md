# exp026_synthesis_orchestrator_core

## 目的

7600を見据え、ad hocなtemplate追加ではなく、phase-awareなprogram synthesis queueを作る。

## 入力

- base: `experiments/exp023_graph_surgery_exp016`
- backlog: `experiments/exp022_program_synthesis_pipeline_7600/synthesis_backlog.csv`

## 結果

- base local estimate: `6479.398825`
- 6500まで: `20.601175`
- 7600まで: `1120.601175`
- `synthesis_queue.csv`: 400 task
- `worker_task_queue.csv`: 上位80 task
- `program_schema.json`: program candidate / lowering / validation / proof log schema
- `lowering_guardrails.csv`: 事前rejectするlowering pattern

## 解釈

Phase 1の上位は `task286`, `task187`, `task198`, `task203`, `task398`, `task313`, `task255`。
まず6500 bridgeを閉じ、その後lookup依存をDSLへ置換して7000/7400/7600へ進む。

## リスク

- leakage risk: high。現baseはsignature lookup由来のlocal upper boundを含む。
- overfitting risk: high。full arc-genとfamily holdoutが必要。

## 次

`exp028_lookup_to_rule_miner` と `exp029_crop_object_synthesizer` を優先する。
