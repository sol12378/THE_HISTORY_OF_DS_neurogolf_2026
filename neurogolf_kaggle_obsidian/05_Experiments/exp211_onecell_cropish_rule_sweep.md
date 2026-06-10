# exp211_onecell_cropish_rule_sweep

## 目的

1x1 cropish候補に単純aggregate rule minerを横展開する。

## 結果

- full_hits: `[]`
- task355 best: `br 70/267`
- task346 best: `bbox_least/least_nz 263/267`
- task291 best: `bbox_mode 37/265`
- task048 best: `bbox_mode 161/270`

## 判断

task346が近い。task346の4 failをrank-switchとして監査する。
