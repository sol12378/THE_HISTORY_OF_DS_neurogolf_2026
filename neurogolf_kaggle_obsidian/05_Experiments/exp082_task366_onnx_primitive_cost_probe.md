# exp082_task366_onnx_primitive_cost_probe

## 目的

`task366` object-marker copy loweringに使うONNX primitiveの公式costを測る。

## 結果

| primitive | cost | notes |
|---|---:|---|
| identity | 0 | baseline sanity |
| static_vertical_bottom_slice_pad | 18021 | 30x30 full-grid Slice+Pad |
| static_horizontal_right_slice_pad | 18021 | 30x30 full-grid Slice+Pad |
| marker_mask_conv_tile | 8115 | non-bg maskを10chへTile |
| where_overlay_proxy | 18001 | full-grid Where + 30x30 zero initializer |

## 判断

30x30 full-grid primitiveは単体でも重すぎる。task366をcost<=250〜600級へ落とすには、full-grid `Slice+Pad` / `Tile` / `Where` を積み上げるrouteは不採用。

## Next

- small patch / sparse coordinate loweringを検討する。
- output全体をmaterializeするmaskではなく、必要object patchだけを低costに生成する方法を探す。
- correctness-first full-grid ONNXはstrict cost `836880` より改善する可能性はあるが、ユーザー目標の600級とはズレるため優先しない。

## Risk

- leakage risk: low。task labelは埋め込んでいない。
- overfitting risk: low。cost-only診断。
