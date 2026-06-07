# exp128_compiler_campaign_30_final_review

## Purpose

30実験compiler campaignの最終レビュー。目的は、革新的compiler設計でcostを下げLB 7500を狙うことだった。30実験に達したため、ここで実行を止める。

## Local Estimate

- initial submit-safe local: `6282.812218`
- final working local: `6282.965236`
- #1〜#30 delta: `+0.153018`
- #21〜#30 delta: `+0.029079`
- #26〜#30 delta: `+0.015026`

LB 7500は未達。現在のLB bestは `5930.40` のまま。exp127 bundleはsubmit-safe micro calibration候補だが、LB7500候補ではない。

## What Worked

有効だったのは、full-validation-gated focused graph surgeryだけだった。

- exp109: `+0.023085`
- exp110: `+0.086887`
- exp111: `+0.013967`
- exp122: `+0.014053`
- exp124: `+0.004776`
- exp125: `+0.004548`
- exp126: `+0.002847`
- exp127: `+0.002854`

合計 `+0.153018`。これはLB calibration / submit-safe post-passとしては価値がある。

## What Failed

- 汎用ONNX optimizer / ORT optimizerはlocalを動かさなかった。
- 単純archetype compiler (`channel_gather`, `static_slice_pad`, `crop+colormap`) はaccepted 0。
- P0 fixed index / bbox anchor / color anchorはfit 0。
- task185 fresh loweringはcore costを下げたが、dynamic selectorが高cost。
- task365/task366 artifact suffix extractionは最終output意味を保持しておらず不発。
- task366 sparse/small-patch writebackも標準ONNXでは重すぎる。

## Why LB 7500 Was Not Reached

今回の改善は既存artifactの冗長node削りに偏った。これは安全だが、1 taskあたりのscore gainが小さい。

LB7500には、全400 taskの多くをcost `250〜600`級へ置き換える必要がある。今回のcampaignでは、そのための新規低cost program数を増やせなかった。

## Compiler Design Lesson

次に必要なのは「ONNXを先に作って試す」方式ではなく、NeuroGolf専用IRで先にcostを見積もってからemitするcompiler。

採用するprimitive:

- one-node / fused data movement
- small `Slice/Gather/Pad`
- small local-mask `Conv`
- safe graph surgery post-pass

事前reject:

- full-grid `Where`
- full-grid or large `GatherND`
- `ScatterND` writeback
- dynamic `MatMul`
- raw coordinate/key lookup

## Stop

campaign #30に到達したため、このcampaignの実験実行はここで停止する。
