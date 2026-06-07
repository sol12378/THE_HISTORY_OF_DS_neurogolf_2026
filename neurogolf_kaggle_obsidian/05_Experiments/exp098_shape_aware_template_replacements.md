# exp098_shape_aware_template_replacements

## 目的

`neurogolf_templates.zip` 由来のPython-grid one-node hitを、公式padding semanticsに合わせた `Slice -> transform -> Pad` に変換して、cost 250〜600級または現行submit-safe artifact改善になるか検証する。

## 結果

- evaluated candidates: `5`
- accepted tasks: `[]`
- accepted delta: `0.000000`
- new local estimate: `6282.812218`

| task | variant | result | cost |
|---:|---|---|---:|
| 150 | flip_lr | skipped: variable shapes `3x3`〜`9x9` | - |
| 155 | flip_ud | skipped: variable shapes `4x4`〜`8x8` | - |
| 087 | shape-aware rot180 | `266_pass_0_fail`, no cost gain | 740 vs 368 |
| 140 | shape-aware rot180 | `265_pass_0_fail`, no cost gain | 740 vs 368 |
| 380 | shape-aware rot90_ccw | rejected: mismatch example 0 | - |

## 解釈

padding mismatch自体は、固定3x3のrot180では `Slice -> rot180 -> Pad` により解消できた。しかしcostは `740` で、既存artifact `368` より悪い。可変shapeのtask150/155は静的ONNXでshape branchが必要になり、単純なshape-aware transformでは提出候補にならない。

## Decision

zip由来template routeは現時点でlocal estimate改善なし。公式one-hotで1ノードにできる場合だけtaxonomyとして使い、可変shapeのflip/rotを250〜600級にする主戦力にはしない。

## Risk

- leakage risk: low。純粋な幾何変換のみ。
- overfitting risk: low〜medium。固定shape仮定に依存するが、全arc-gen validationで判定済み。
