# exp098_shape_aware_template_replacements

## 目的

`neurogolf_templates.zip` 由来のone-node template hitのうち、公式padding semanticsで失敗したflip/rot候補を、active area `Slice -> transform -> Pad` に変えて250〜600級へ入るか検証する。

## 結果

- evaluated candidates: `5`
- accepted tasks: `[]`
- accepted delta: `0.000000`
- new local estimate: `6282.812218`

## 解釈

全gridを直接flip/rotするのではなく、固定active shapeを切り出してから変換すればpadding mismatchは避けられる。ただし、`Slice+transform+Pad` の中間tensor costが支配するため、250〜600級に入るかはshape面積に強く依存する。

## Decision

現行submit-safe artifactより安く、かつfull validationを通ったものだけ採用候補にする。改善がなければzip由来template routeは、さらに小さいshapeまたは既存artifact surgery対象に限定する。

## Risk

- leakage risk: low。
- overfitting risk: fixed shapeに依存するためlow〜medium。train/test/arc-gen full validationを通してから判断する。
