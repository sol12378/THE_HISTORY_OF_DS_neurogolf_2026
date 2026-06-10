# exp129_compiler_farm_core

## Hypothesis

NeuroGolf専用IRとcost extractorをONNX生成前に置けば、full-grid中間や`ScatterND`/`Where`型の高cost候補を事前rejectでき、実験farmをscore-producing候補へ集中できる。

## Result

- 実装: `experiments/neurogolf_farm/`
  - `ir.py`: task-level IR (`IRProgram`, `IRNode`, `PrimitiveKind`)
  - `cost_extractor.py`: high-cost pattern guardrail
  - `public_code.py`: public CODE registry / 6285 floor gate
  - `bundle_manager.py`: accepted candidate ledger / bundle gate
  - `farm_runner.py`: smoke orchestration
- smoke:
  - `smoke_channel_gather`: `250-600_plausible`
  - `smoke_slice_pad`: `250-600_plausible`
  - `smoke_full_grid_where`: hard reject
  - `smoke_scatter_writeback`: hard reject
- public CODE registry:
  - 既存公開CODE/blend系実験をregistry化。
  - best observed localは `exp_b036_beicicc_full_arc_blend` の `6276.160646`。
  - Kaggle LB `6285+` の証拠はregistry内にないため、6285 floor readyは `False`。
- bundle ledger smoke:
  - full-arc pass / cost改善 / low riskのみacceptedに残ることを確認。

## Decision

公開CODEは提出下限ではなくteacher/intelligenceとして取り込む。full-arc validationとKaggle LB evidenceが揃った候補だけを6285 floorへ昇格する。直接blendで下限を宣言するのはpublic LB overfit / leakage riskが高い。

## Next

1. 追加public notebookのURL/sourceをregistryへ入力する。
2. 6285 claimed候補ごとにfull-arc validation、cost extraction、LB calibrationを通す。
3. IR/cost extractor/bundle ledgerをcandidate evaluation pipelineに接続し、ONNX emission前rejectとaccepted candidate管理を標準化する。
