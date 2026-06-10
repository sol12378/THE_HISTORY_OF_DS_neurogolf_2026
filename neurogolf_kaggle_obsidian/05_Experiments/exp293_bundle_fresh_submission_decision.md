# exp293_bundle_fresh_submission_decision

## Plan

real candidate manifest 生成後に、現行best lineage除外と提出判定を1回で実行できる helper を作る。

## Do

`bundle_manager.py` に `BundleLedger.fresh_submission_decision()` を追加。

## Check

- 履歴全体: `accepted_count=104`, `should_submit=true`
- fresh-only: `accepted_count=0`, `candidate_estimate=6008.9`, `should_submit=false`

## Act

提出なし。fresh local estimate改善なし。
