# exp108_compiler_campaign_10_review

## Purpose

compiler campaign #10として、#1〜#10のlocal estimate変化を測定し、LB 7500へ向けてこの10実験が有意義だったかを問い直す。

## Measurement

- base local estimate: `6282.812218`
- new local estimate after #10: `6282.812218`
- local delta #1〜#10: `0.000000`
- local delta #6〜#10: `0.000000`
- accepted task count: `0`

## Review

結論は厳しい。#1〜#10はlocal/LBを全く動かしていないため、LB向上に直接有効だったとは言えない。

ただし、以下の否定知識は有効だった。

- 汎用OSS/ORT optimizer tuningは主戦力にならない。
- simple `channel_gather` / `static_slice_pad` / `crop+colormap` rolloutは現行bestに吸収済み。
- P0小出力taskは絶対座標static index mapではない。
- 1x1 P0 taskは全bbox/min色bbox anchor近傍の単純抽出ではない。
- task185は局所logic bypassではcostが下がらない。

## Decision

#11〜#15は広いanchor scanを止める。各実験は、既にPython ruleが解けた高gain task、または低cost artifactから構造が見えているtaskに対して、task-specific fused compilerを作る。

優先順位:

1. task185: 4x4 lattice homogeneous-2x2 ruleのfused lowering / subgraph extraction。
2. task048: bridge connectivityのclosed-form特徴または既存artifact surgery。
3. 低cost artifactからのsubgraph template extraction。特に `computed_slice_pad`, `tiny_dynamic_shape_index`, one-node Conv。

## Risk

- leakage risk: review自体はlow。
- overfitting risk: 次blockはtask-specificになるためmedium。visual rule説明、full arc-gen validation、single-task LB calibrationで抑える。
