# exp_b014_task037_fullarc_graph_surgery

## 目的

task037 strict artifactを全arc-gen gate付きでgraph surgeryし、b010で見つかったruleに近い既存実装を安全に削れるか確認する。

## 結果

- node count: 27
- initializer count: 13
- candidate count: 22
- status counts: {'rejected': 19, 'no_cost_gain': 3}
- accepted: None
- local estimate delta: 0.000000

## 判断

改善があればsingle-task deltaとしてKaggle較正候補。改善がなければ、既存strict artifact surgeryではなくrule-specific loweringへ戻る。
