# exp044_task031_dynamic_bbox_lowering

## 目的

exp042最大hitである task031 `bbox_nonzero` を、動的bbox検出ONNXとして実装し、既存artifactを丸ごと置換できるか検証する。

## 結果

- status: no_gain
- validation: 266 pass / 0 fail
- baseline cost: 18,616
- candidate cost: 1,050,228
- local delta: +0.000000

## 解釈

ruleは正しい。しかし `ArgMax + GatherND` で30x30全channelの動的indexを作るloweringは、巨大な中間テンソルとinitializerが支配して完全に負ける。

## 判断

full-grid `GatherND` bbox cropはguardrail入り。object/bbox compilerは、行/列単位の小さい制御、既存artifactからのsurgery、または形状候補の静的分岐をより小さく表現する方向が必要。

## 次

既存のtask031 artifactをprofileして、18,616 costでbbox cropをどう実現しているかを読む。そこから削るか、同じ構造をより小さく再合成する。
