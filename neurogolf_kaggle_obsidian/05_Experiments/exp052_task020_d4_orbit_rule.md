# exp052_task020_d4_orbit_rule

## 目的

task020を5x5 bbox中心まわりのD4対称orbit補完として説明できるか検証する。

## 結果

- D4 orbit count: 6
- pass: 160/266
- train: 3/3 pass
- test: 1/1 pass
- arc-gen: 156/262 pass
- failure: candidate_count=0 が106件

## 解釈

D4対称性はtask020の大きな部分を説明するが、まだ全体ruleではない。残りは別のlocal coordinate frame、別orbit、または色role条件が必要。

## 次

失敗例のtarget positionsを抽出し、D4 orbit以外のfinite templateを分類する。
