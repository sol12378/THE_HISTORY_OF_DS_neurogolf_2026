# exp216_onecell_cropish_wide_rule_sweep

## 目的

exp087の1x1 cropish候補全体へ、安くloweringできる可能性のある入力集約ruleを横展開する。

## 仮説

1x1出力taskには、mode/least/corner/bbox/edge/interiorのような小さな集約で説明できるものが残っている。full hitが見つかれば、one-cell scalar graphとして低cost化できる可能性がある。

## 結果

- evaluated: `[48, 56, 103, 291, 346, 355]`
- full_hits: `[]`
- near_hits(<=10 fail): `[(346, 'interior_least_nz', 2)]`
- best: `[(48, 'all_second_least_nz', 172, 270), (56, 'all_least_nz', 5, 46), (103, 'all_least_all', 0, 223), (291, 'middle_col_least_nz', 61, 265), (346, 'interior_least_nz', 265, 267), (355, 'all_second_least_nz', 100, 267)]`

## 判断

full hitがあればtiny ONNX cost probeへ進める。near hitは失敗監査を行う。full/near hitが薄ければ、1x1単純集約laneは優先度を下げ、component-freeな別small-output familyへ移る。

## リスク

- leakage risk: low。入力だけの集約rule。
- overfitting risk: medium-low。単純ruleのため低いが、near hit補正で分岐を足す場合は上がる。
