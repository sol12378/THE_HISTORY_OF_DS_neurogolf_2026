# exp221_fixed_smallshape_crop_sweep

## 目的

固定小shape cropish候補に、static Slice化しやすい固定anchor cropと診断用bbox cropを横展開する。

## 結果

- full hits:
  - task039 `3x3` bbox crop, baseline cost `7772`
  - task135 `3x3` fixed top-right crop, baseline cost `360`
  - task326 `2x2` fixed/bbox top-left crop, baseline cost `160`
- useful fixed-anchor high-cost hit: none

## 判断

task039はdynamic crop cost wallで保留。task135/326はbaselineが低く、提出候補にしない。

## リスク

- leakage risk: low
- overfitting risk: medium-low
