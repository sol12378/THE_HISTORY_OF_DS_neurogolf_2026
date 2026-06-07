# exp123_compiler_campaign_25_review

## Purpose

compiler campaign #21〜#25のlocal estimate変化と、LB 1位方針への有意義性をレビューする。

## Result

- #21〜#25 local delta: `+0.014053`
- #1〜#25 local delta: `+0.137992`
- new local estimate: `6282.950210`
- score-producing experiment: exp122のみ

## Verdict

`artifact_harvesting_failed_but_postpass_still_useful`

task365/task366のsuffix extraction / cast suffix routeは不発。中間tensorは最終shapeに近くても最終意味を持っていない。

focused surgeryはまだ効くがmicro-deltaであり、LB 1位を狙うには桁が足りない。

## Next Policy

#26〜#30は以下に絞る。

- one-node / fused data movement
- small `Slice/Gather/Pad`
- small local-mask `Conv`
- safe graph surgery post-pass
- full-grid `Where/GatherND/ScatterND` とdynamic `MatMul` は事前reject

OSSは丸ごと輸入ではなく、低cost表現とrewrite/e-graph発想をNeuroGolf専用compilerへ移植する。
