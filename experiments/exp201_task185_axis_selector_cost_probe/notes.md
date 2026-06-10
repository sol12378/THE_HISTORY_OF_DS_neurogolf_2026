# exp201_task185_axis_selector_cost_probe

## 目的

exp200で成立したtask185 axis-separable selectorについて、pairwise window scoring proxyより安いONNX cost floorになるか測る。

## 結果

- best_scored: `axis_conv_score_proxy`
- best_cost: `4156`
- pairwise_static_proxy_cost_from_exp117: `76194`

## 判断

axis selectorが十分安ければ、次はcorrectness-first task185 loweringへ進む。baseline `59584` を超えるなら、selectorは発見済みでもscore laneとしては弱い。

## リスク

- leakage risk: low。cost proxyのみ。
- overfitting risk: medium-low。提出candidateではない。
