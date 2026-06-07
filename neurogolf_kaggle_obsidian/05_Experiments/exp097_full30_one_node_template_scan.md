# exp097_full30_one_node_template_scan

## 目的

`exp096` で可変サイズgridのpadding mismatchが判明したため、全例が30x30のtaskだけに限定して1ノード公式templateをscanする。

## 結果

- full30 task count: `3`
- full30 tasks: `54`, `74`, `255`
- full hits: `0`

## 解釈

full-grid 1ノード `Slice` / `Transpose` / channel `Gather` はcostとしては強いが、適用可能な全例30x30 taskが少なく、該当3件にも単純template hitはなかった。

## Decision

zip由来のfull-grid 1ノードtemplateは主戦力ではない。shape-aware crop/transform/pad、既存artifact surgery、または小出力closed-form ruleへ戻る。

## Risk

- leakage risk: low。
- overfitting risk: low。
