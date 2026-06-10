# exp277_farm_submission_decision_helper

## 仮説

提出可否を各実験が手作業で判断すると、重複taskや精度gate落ち候補のdeltaを誤って足すリスクがある。`best_total_local_delta` を使った helper を ledger に持たせれば、提出ポリシーを実装に固定できる。

## 実施

- 変更対象は `experiments/neurogolf_farm/bundle_manager.py` のみ。
- `BundleLedger.submission_decision(submitted_best_estimate)` を追加。

## 結果

- deltaあり: `6008.90 + 0.2 -> should_submit=true`
- 精度gate落ち: `6008.90 + 0 -> should_submit=false`
- farm smoke成功。

## 判断

提出なし。tooling helper追加のみで、候補bundleのlocal estimate更新ではない。

## 次アクション

farm smoke/result に `submission_decision` を標準出力し、提出可否判定を実験artifactへ残す。
