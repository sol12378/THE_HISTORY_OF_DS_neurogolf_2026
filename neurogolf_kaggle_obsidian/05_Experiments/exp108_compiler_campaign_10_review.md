# exp108_compiler_campaign_10_review

## Purpose

compiler campaign #10。#1〜#10のlocal estimate変化を測定し、LB 7500へ向けた有効性を問い直す。

## Measurement

- base local estimate: `6282.812218`
- new local estimate: `6282.812218`
- local delta #1〜#10: `0.000000`
- local delta #6〜#10: `0.000000`
- accepted task count: `0`

## Review

LB向上への直接有効性は低い。10実験でlocal estimateが一切動いていないため、このまま同じ探索を続けるのは不適切。

有効だった点は、否定知識として以下を確定したこと。

- 汎用OSS/ORT optimizer tuningは主戦力にならない。
- simple archetype rolloutは現行bestに吸収済み。
- P0小出力は絶対座標static index mapではない。
- 1x1 P0は全bbox/min色bbox anchor近傍の単純抽出ではない。
- task185の局所logic bypassはcostを下げない。

## Decision

#11〜#15はtask-specific fused compilerへ切り替える。

優先lane:

1. task185 lattice homogeneous-2x2 fused lowering / subgraph extraction。
2. task048 bridge connectivityのclosed-form feature / surgery。
3. low-cost artifact subgraph extraction (`computed_slice_pad`, `tiny_dynamic_shape_index`, one-node Conv)。

広いanchor scanは、視覚ruleが先にある場合だけ行う。

## Risk

- leakage risk: reviewはlow。
- overfitting risk: 次blockはtask-specificになるためmedium。visual rule説明、full arc-gen validation、single-task LB calibrationを必須にする。
