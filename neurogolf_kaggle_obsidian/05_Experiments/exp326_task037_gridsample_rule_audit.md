# exp326 task037 GridSample rule audit

## 目的

exp325 採点待ち中の独立作業。GridSample queue の次候補 task037 が、task251 より実装しやすい値コピー型かを確認する。

## 結果

- rule: `opposite_ray_diag_same_color`
- validation: `266_pass_0_fail`
- changed cells mean: `8.466165413533835`
- output new color examples: `0`
- generated new color cells: `0`
- max endpoint distance hist: `{1: 255, 2: 544, 3: 697, 4: 506, 5: 250}`

## 判断

task037 は出力で新しい非ゼロ色を作らず、対角方向の既存 endpoint 色をコピーしている。task251 のような color1 生成問題ではないため、GridSample/shift lowering の次候補としてより自然。

過去の naive Conv visibility は cost `441976` で負けているため、次は full-grid Conv ではなく、距離1〜5の bounded diagonal shifts または GridSample samples と equality mask の合成を試す。

## Submission

提出なし。audit のみ。

## Risk

- leakage risk: low。local examples 上の説明可能 rule 再監査のみ。
- overfitting risk: low-to-medium。rule は full-arc valid だが、ONNX candidate は別途 cost と Kaggle calibration が必要。

## Next

exp327 として task037 bounded diagonal shift / GridSample equality mask candidate を作る。
