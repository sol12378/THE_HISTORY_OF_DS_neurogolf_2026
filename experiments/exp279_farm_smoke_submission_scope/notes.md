# exp279_farm_smoke_submission_scope

## 目的

exp278で farm smoke/result に `submission_decision` を表示したが、smoke dummy ledger の `should_submit=true` が実提出判断と紛らわしい。smoke専用の判定であることを result に明示する。

## 変更

- `experiments/neurogolf_farm/farm_runner.py` の `bundle_smoke` に以下を追加した。
  - `submission_decision_scope`
  - `submission_decision_reason`

## 確認

- `py_compile` 成功。
- `FarmRunner.run_smoke()` 成功。
- exp279では `submission_decision_scope=smoke_dummy_not_kaggle_candidate`。
- reason は `run_smoke uses synthetic ledger rows only; do not submit from this decision`。

## 判断

提出なし。これは smoke result の誤読防止であり、実候補bundleのlocal estimate更新ではない。

## リスクと次アクション

- leakage risk: なし。
- overfitting risk: なし。
- farm tooling の主要ギャップは一通り埋まった。次は実候補生成側に戻り、`submission_decision` を使って提出可否を記録する。
