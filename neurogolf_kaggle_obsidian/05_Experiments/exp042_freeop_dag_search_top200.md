# exp042_freeop_dag_search_top200

## 目的

既存artifactを削る局所surgeryではなく、既知solver/DSL探索で最小回路を新規合成する方針が本当に有効かを検証する。

## 仮説

少数でも、軽量DSL programが train/test/all arc-gen に完全一致すれば、1taskあたりのproxy利得はgraph surgeryの微小改善より大きい。

## 実験

- base: `experiments/exp023_graph_surgery_exp016`
- 対象: 全400 task
- 検証: train + test + all arc-gen 完全一致
- primitive: flip/rotate/transpose, color LUT, fixed crop, nearest upscale, constant output, nonzero/color bbox crop
- 出力: `experiments/exp042_freeop_dag_search_top200/result.json`, `program_hits.csv`, `scan_summary.csv`

## 結果

- full-arc exact program hits: 9
- total proxy delta: +16.2291
- exp040 deeper surgery delta: +0.0509
- proxy upsideは exp040 surgery の約319倍
- 主なhit:
  - task031: `bbox_nonzero`, proxy +3.5935
  - task150: `flip_h`, proxy +2.3573
  - task155: `flip_v`, proxy +2.3573
  - task223: `nearest_upscale_3`, proxy +2.2380
  - task380: `rot90_ccw`, proxy +1.7383

## 解釈

最初の狭い top200/global-transform scan はhitゼロだったため、「浅いFREE opだけで高cost taskを解く」という仮説は否定された。一方で、全400へ広げてobject/bbox primitiveを入れると、全arc-gen一致の低cost programが9件見つかった。

これは、6500突破の主戦略を「artifact surgery継続」から「DSL programを見つけ、ONNX loweringで丸ごと置換する」へ移す根拠になる。ただし現時点ではproxy costであり、提出候補ではない。

## リスク

- leakage risk: 低〜中。signature lookupは使っていないが、all arc-genをproofに使った。
- overfitting risk: 中。program familyは単純だが、ONNX lowering後にofficial utilityで再検証が必要。

## 次

exp043で固定flip/rot/crop/upscale hitからONNX loweringを実装し、実costを測る。bbox_nonzeroは動的bbox loweringが必要なので第2段に回す。
