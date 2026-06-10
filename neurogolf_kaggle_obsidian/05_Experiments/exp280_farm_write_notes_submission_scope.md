# exp280_farm_write_notes_submission_scope

## 仮説

exp279で result には smoke-only の `submission_decision_scope/reason` を出したが、`write_notes()` には出ていなかった。notesにも同じ情報を出せば、smoke dummy の `should_submit=true` を実提出判断と混同しにくくなる。

## 実施

- 変更対象は `experiments/neurogolf_farm/farm_runner.py` のみ。
- `FarmRunner.write_notes()` に `submission decision scope` と `submission decision reason` を追加。

## 結果

- `py_compile` 成功。
- `FarmRunner.run_smoke()` + `write_notes()` 成功。
- generated notes に以下を確認:
  - `submission decision scope: smoke_dummy_not_kaggle_candidate`
  - `submission decision reason: run_smoke uses synthetic ledger rows only; do not submit from this decision`

## 判断

提出なし。notes可視化の tooling 較正のみで、実候補bundleのlocal estimate更新ではない。

## 次アクション

farm tooling の提出判定系は整った。次は実候補生成・評価へ戻り、`submission_decision` を実候補 result に使う。
