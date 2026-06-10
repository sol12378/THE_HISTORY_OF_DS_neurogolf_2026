# exp211_onecell_cropish_rule_sweep

## 目的

1x1 cropish候補にtask355の単純集約rule minerを横展開し、短いruleの当たりを探す。

## 結果

- full_hits: `[]`
- best: `[(355, 'br', 70, 267), (346, 'bbox_least', 263, 267), (291, 'bbox_mode', 37, 265), (48, 'bbox_mode', 161, 270)]`

## 判断

full hitがあればtiny ONNX化。なければこの単純aggregate laneは打ち切り、別rule familyへ移る。
