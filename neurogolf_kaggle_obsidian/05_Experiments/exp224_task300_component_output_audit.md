# exp224_task300_component_output_audit

## 目的

task300について、出力がどの入力componentのcrop/maskに対応するか、単純selectorで説明できるかを監査する。

## 結果

- baseline cost: `77546`
- `crop_exact`: `267/267`
- `mask_colorized`: `267/267`
- selector: largest component by cell count (`rank_size_desc=0`) が `267/267`
- ambiguous_or_no_exactish: `0`

## 判断

task300は「最大size 4-connected same-color componentのbbox crop」で解ける。大きなsolved-rule assetとして次はcost proxyへ進める。

## リスク

- leakage risk: low
- overfitting risk: medium-low
- lowering risk: high。component selectionとdynamic bbox cropが重くなりやすい。
