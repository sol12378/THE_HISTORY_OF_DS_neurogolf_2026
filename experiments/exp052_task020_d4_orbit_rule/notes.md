# exp052_task020_d4_orbit_rule

## 仮説

task020は5x5 bbox中心まわりのD4 orbit completionで説明できる。

## 結果

- status: partial
- pass: 160/266
- split pass: {'train': 3, 'test': 1, 'arc-gen': 156} / {'train': 3, 'test': 1, 'arc-gen': 262}
- fail reasons: {'candidate_count=0': 106}

## 解釈

train template暗記ではなく、D4対称性から候補orbitを全列挙した。full passならsignature lookupを明示ruleへ置換できる。
