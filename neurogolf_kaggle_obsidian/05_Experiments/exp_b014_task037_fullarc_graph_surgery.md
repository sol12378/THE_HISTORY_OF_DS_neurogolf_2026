# exp_b014_task037_fullarc_graph_surgery

## 目的

task037 strict artifactを全arc-gen gate付きでgraph surgeryし、b010で見つかったruleに近い既存実装を安全に削れるか確認する。

## 結果

- baseline: `exp005_top_cost_rewrite_strict`
- task037 strict cost: `63726`
- node count: `27`
- initializer count: `13`
- op counts: `Relu 7`, `Conv 6`, `Mul 5`, `Slice 2`, 他
- generated candidates: `22`
- status counts:
  - rejected: `19`
  - no_cost_gain: `3`
- accepted: none
- local estimate delta: `0.0`
- submission: `no_submit`

## 解釈

strict artifactはすでに27 node程度まで圧縮されており、単純bypassでは実costが下がらない。3件のbypassは `266_pass_0_fail` だったが、costは `63726` のままだった。

## Decision

task037はgraph surgeryではなく、rule-specific loweringを再設計する。full-grid Conv visibilityは高すぎるため、diagonal-specific mask / triangular Gather / precomputed diagonal basis のような小型表現を探索する。
