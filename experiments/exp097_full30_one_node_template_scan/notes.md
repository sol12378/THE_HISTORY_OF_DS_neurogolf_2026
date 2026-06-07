# exp097_full30_one_node_template_scan

## 目的

`exp096` で可変サイズgridのpadding mismatchが判明したため、全例が30x30のtaskだけに限定して1ノード公式templateをscanする。

## 結果

- full30 task count: `3`
- full hits: `0`

## Decision

full30 hitのみ、full-grid one-node Slice/Transpose/Gather replacementへ進める。
