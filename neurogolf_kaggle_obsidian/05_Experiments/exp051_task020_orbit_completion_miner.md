# exp051_task020_orbit_completion_miner

## 目的

train例から得た相対位置orbit templateでtask020を説明できるか検証する。

## 結果

- train templates: 3
- pass: 96/266
- train: 3/3 pass
- arc-gen: 93/262 pass

## 解釈

train template暗記では不足。arc-genにはtrainに出ていないorbit familyがあるため、幾何対称性から汎化する必要がある。
