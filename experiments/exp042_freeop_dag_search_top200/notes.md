# exp042_freeop_dag_search_top200

## 仮説

既存artifactを削る局所surgeryではなく、軽量DSL/DAG探索で最小回路を新規合成すると、少数の完全一致hitだけでも exp040 の追加surgery幅を大きく上回る。

## 実験

- base: `experiments\exp023_graph_surgery_exp016`
- 対象: base cost上位 400 task
- 検証: train + test + all arc-gen に完全一致する lightweight program のみ採用
- 探索primitive: flip/rotate/transpose, color LUT, fixed crop, nearest upscale
- 今回は proof experiment として proxy cost で評価し、ONNX lowering は次段に残した。

## 結果

- improved program hits: 9
- total proxy delta: 16.229076
- exp040 deeper surgery delta: 0.050856
- multiple vs exp040 surgery: 319.12x

## 解釈

hitが少数でも、丸ごと小さいprogramへ置換できる候補は1taskあたりの利得が局所surgeryより桁違いに大きい。これは「artifactを削る」より「既知変換を最小回路として新規合成する」方針が、6500突破の主戦略として正しいことを支持する。

## リスク

- leakage risk: 低〜中。signature lookupは使っていないが、all arc-gen を proof に使っているため、提出候補化では公式utilityによる再検証が必要。
- overfitting risk: 中。primitive familyが狭いので過剰な探索ではないが、proxy cost段階でありONNX実costは未確定。

## 次アクション

1. top hit family の ONNX lowering を実装する。
2. `program_hits.csv` の上位から exp043 で実cost計測する。
3. 実costがproxyより悪化するfamilyを guardrail に追加する。
