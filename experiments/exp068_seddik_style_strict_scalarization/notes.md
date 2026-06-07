# exp068_seddik_style_strict_scalarization

## 目的

Seddik notebookの外科的ONNX圧縮を、現在のsubmit-safe base `exp066` にfull-arc gated post-passとして適用する。

## 結果

- targeted tasks: 37
- improved tasks: [383, 284, 110, 117, 17, 77, 212, 44, 358, 178, 267, 20, 80, 58, 128, 36, 144, 65, 26, 357, 248, 296, 161, 134, 51, 370, 182, 114, 245, 115, 45, 133, 206, 157]
- local delta: 0.170390656
- new local estimate: 6282.610350375
- Kaggle submission: ref `53417598`, LB `5930.27`
- LB delta vs exp066: `+0.17`

## 手法

- unused initializer prune
- exact duplicate initializer dedup
- uniform initializer scalarization for broadcast-safe consumer ops

## Decision

micro-deltaとして提出し、local delta `+0.170391` とLB delta `+0.17` が一致した。Seddik-style post-passはstrict-safeな小改善として継続利用できる。
