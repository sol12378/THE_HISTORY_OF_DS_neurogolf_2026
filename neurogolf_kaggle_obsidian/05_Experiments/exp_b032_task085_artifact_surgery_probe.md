# exp_b032_task085_artifact_surgery_probe

## 目的

task085既存artifactの明らかな `Cast` を削れるか確認する。

## 結果

- base: `exp_b025_submit_safe_delta_union`
- task: `85`
- candidates: `2`
- valid: `0`

| candidate | status | reason |
|---|---|---|
| `cast_bypass_keep_node` | rejected | ORT type error |
| `cast_bypass_remove_node` | rejected | ORT type error |

`Cast` をbypassすると、内部 `Slice` / downstream graphがfloat16を期待しているのにfloat inputが流れ、ORT loadに失敗する。

## Decision

task085 artifactはfloat16内部計算に依存しており、単純Cast削除は不可。ruleは解けているが既存artifactはcompactなので、短期score目的では別laneへ移る。
