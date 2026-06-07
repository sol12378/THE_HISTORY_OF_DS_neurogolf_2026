# exp051_task020_orbit_completion_miner

## 仮説

task020は、trainから得られる色ごとの相対位置orbit templateを補完するruleで説明できる。

## 結果

- status: partial
- templates: 3
- pass: 96/266
- split pass: {'train': 3, 'arc-gen': 93} / {'train': 3, 'test': 1, 'arc-gen': 262}

## 解釈

train由来templateだけでtest/all arc-genへ汎化するかを検証した。通ればsignature lookupではなく、人間可読なorbit completion ruleとして次にONNX化する。
