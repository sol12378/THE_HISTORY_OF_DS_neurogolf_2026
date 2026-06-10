# exp281_farm_write_notes_submission_decision_values

## 仮説

`write_notes()` に submission decision の scope/reason は出るようになったが、`candidate_estimate` と `should_submit` が出ていないと、提出可否の数値根拠を notes だけで確認できない。

## 実施

- 変更対象は `experiments/neurogolf_farm/farm_runner.py` のみ。
- `FarmRunner.write_notes()` に以下を追加。
  - `submitted_best_estimate`
  - `candidate_estimate`
  - `should_submit`

## 結果

- `py_compile` 成功。
- `FarmRunner.run_smoke()` + `write_notes()` 成功。
- generated notesに以下を確認:
  - `submitted best estimate: 6008.9`
  - `candidate estimate: 6009.0`
  - `should submit: True`

## 判断

提出なし。これは smoke dummy notes の可視化であり、実候補bundleのlocal estimate更新ではない。

## 次アクション

farm tooling 整備を一旦止め、実候補生成へ戻って `submission_decision` を本物の候補 result に適用する。
