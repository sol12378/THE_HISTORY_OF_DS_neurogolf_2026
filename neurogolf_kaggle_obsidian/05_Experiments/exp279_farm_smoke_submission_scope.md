# exp279_farm_smoke_submission_scope

## 仮説

smoke dummy ledger の `should_submit=true` は、実候補bundleの提出可否と混同される危険がある。scope/reasonを明示すれば、提出判断ミスを防げる。

## 実施

- 変更対象は `experiments/neurogolf_farm/farm_runner.py` のみ。
- `bundle_smoke.submission_decision_scope` と `submission_decision_reason` を追加。

## 結果

- `submission_decision.should_submit=true`
- `submission_decision_scope=smoke_dummy_not_kaggle_candidate`
- `submission_decision_reason=run_smoke uses synthetic ledger rows only; do not submit from this decision`
- farm smoke成功。

## 判断

提出なし。tooling誤読防止のみで、候補bundleのlocal estimate更新ではない。

## 次アクション

実候補生成へ戻り、farm result の `submission_decision` を使って提出可否を記録する。
