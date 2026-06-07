# exp101_archetype_compiler_v1

## 目的

exp100で抽出した安全寄りarchetype (`channel_gather`, `static_slice_pad`, `crop+channel_gather+pad`) を全taskに流し、current submit-safe bundleより低costな候補が出るか測る。

## 結果

- generated candidates: `9`
- tasks with candidates: `6`
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.812218`

## 解釈

単純global colormap/common cropは現行bestにほぼ吸収済み。候補が出ても、task016/337のGatherは現行cost `10` と同等、task135/326のSlice+Padは現行artifactより高costだった。

## Decision

`channel_gather` / `static_slice_pad` の単純横展開ではscore-producingにならない。次は `one_node_conv_kernel` と `computed_slice_pad` の中身を掘る。

## Risk

- leakage risk: low-to-medium。
- overfitting risk: medium。採用候補が出た場合はLB較正対象。
