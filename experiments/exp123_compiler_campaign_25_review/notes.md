# exp123_compiler_campaign_25_review

## Hypothesis

#21〜#25は、task365/task366へのpivotがLB 1位に向けて有効だったか、またlocal estimateが更新されたかを測るレビュー枠である。

## Result

- #21 exp119: task365 suffix extractionはlocal delta `0`。
- #22 exp120: task366 direct suffix extractionはlocal delta `0`。
- #23 exp121: task366 Cast suffix extractionもlocal delta `0`。
- #24 exp122: focused surgery三巡目はaccepted 5 task、local delta `+0.014053`。
- #21〜#25 window delta: `+0.014053`。
- campaign #1〜#25 delta: `+0.137992`。
- new local estimate: `6282.950210`。

## Interpretation

task365/task366の既存artifact harvestingは現時点で有効ではない。中間tensorはmask/type/shape的に近くても、最終outputの意味を保持しておらず、suffix cutやCast追加では削れない。

一方、focused artifact surgeryは三巡目でも微小に効いた。これはsubmit-safe post-passとして価値があるが、7500やLB 1位に必要な桁ではない。

## Decision

#26〜#30では、task365/task366のsuffix extractionを止める。次は「OSSを丸ごと輸入する」よりも、低cost artifactから逆輸入したprimitiveをcompilerに組み込む:

- one-node / fused data movement
- small `Slice` / `Gather` / `Pad`
- small local-mask `Conv`
- safe initializer/node surgery post-pass
- full-grid `Where` / `GatherND` / `ScatterND` / dynamic `MatMul` は事前reject

LB較正は、exp122のような小さいofficial-valid deltaを提出候補にして行う。大きなlookup/public artifact bundleはlocal/LB乖離が大きいため主評価にしない。

## Risk

- leakage risk: exp122自体は低い。今後OSS/compiler trickを輸入する場合、public artifact依存やraw table化は中〜高risk。
- overfitting risk: focused surgeryは低〜中。ただし同一task反復なのでLB calibrationが必要。
