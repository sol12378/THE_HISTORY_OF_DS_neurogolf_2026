# exp_b032_task085_artifact_surgery_probe

## 目的

task085既存artifactから明らかなCastを削れるか確認する。

## 結果

- candidates: `2`
- valid: `0`
- reason: Cast bypassすると `safe_name_15` がfloat16期待なのにinput floatが流れ、ORT type error。

## 判断

task085 artifactはfloat16内部計算に依存しており、単純Cast削除は不可。rule hitはあるが、このartifactはかなりcompactなので、短期score目的なら別laneへ移る。
