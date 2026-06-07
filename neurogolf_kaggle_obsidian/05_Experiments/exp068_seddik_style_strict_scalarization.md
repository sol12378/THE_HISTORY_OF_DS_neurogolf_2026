# exp068_seddik_style_strict_scalarization

## 目的

Seddik notebook由来の外科的ONNX圧縮を、`exp066` submit-safe baseへfull-arc gated post-passとして適用する。

## 結果

- targeted tasks: 37
- improved tasks: 34
- local delta: `+0.170391`
- local estimate: `6282.439960 -> 6282.610350`
- Kaggle ref: `53417598`
- LB: `5930.27`
- LB delta vs exp066: `+0.17`

## 手法

- unused initializer prune
- exact duplicate initializer dedup
- uniform initializer scalarization for broadcast-safe consumer ops

## 改善task

`383, 284, 110, 117, 17, 77, 212, 44, 358, 178, 267, 20, 80, 58, 128, 36, 144, 65, 26, 357, 248, 296, 161, 134, 51, 370, 182, 114, 245, 115, 45, 133, 206, 157`

## 解釈

local delta `+0.170391` に対してLB deltaも `+0.17` で一致した。`exp066` に続き、strict-safe official validation経路はKaggle LBとよく合う。一方、gain規模は小さいので、7700へはこのpost-passだけでは足りない。主戦略は引き続き明示rule/compiler replacementである。

## Decision

Seddik-style post-passは今後のbundle生成後に標準適用する。次はKarnak description priorを使い、L1/L2 sparse fillやL4 crop/shape compilerの探索順序を再重み付けする。
