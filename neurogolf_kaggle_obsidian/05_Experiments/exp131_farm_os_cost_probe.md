# exp131_farm_os_cost_probe

## Hypothesis

レビュー指摘に従い、cost extractorを現行cost式寄りの数値proxyへ修正すれば、cheap controlとhigh-risk controlを実測前により妥当に分離できる。

## Result

`cost_extractor`を修正し、最終outputではなく中間value memoryとinitializer parameter countをproxyとして出すようにした。6本のONNX cost probeを実行した。

| case | proxy | official | band |
|---|---:|---:|---|
| identity | 0 | 0 | 250-600_plausible |
| channel_gather | 10 | 0 | 250-600_plausible |
| conv1x1 | 100 | 0 | 250-600_plausible |
| full_grid_where | 18000 | 0 | high_cost_probe_only |
| small_where_expand | 36024 | 36000 | high_cost_probe_only |
| static_slice_pad_3x3 | 381 | 360 | 250-600_plausible |

## Interpretation

memory-heavy intermediateの検出は改善した。特に`small_where_expand`はofficial `36000`であり、従来のnode数ベース判定なら見逃す危険があった。`static_slice_pad_3x3`はproxy `381` / official `360`で600級候補として妥当。

一方、initializer parameter proxyはofficialより過大評価する場合がある。`channel_gather`, `conv1x1`, `full_grid_where`はofficial `0`で、constant/initializerの扱いは公式traceと完全には一致しない。したがってfarmはproxyで候補を並べ、acceptanceは必ずofficial `score_network`とvalidationで行う。

## Decision

farm OSは実測校正付き検問として前進したが、score-producing optimizerではまだない。次はtask別IR -> ONNX emitter -> official score -> full-arc validation -> bundle ledgerを直結する。

## Leakage / Overfitting Risk

task001に対するcost probeであり、出力正解性は評価していない。local estimateやLBへ加算しない。
