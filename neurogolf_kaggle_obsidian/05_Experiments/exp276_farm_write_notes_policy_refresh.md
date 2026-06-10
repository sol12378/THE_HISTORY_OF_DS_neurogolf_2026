# exp276_farm_write_notes_policy_refresh

## 仮説

farm tooling の実装は output bytes / Where MAC=0 / best-by-task delta 方針へ更新済みだが、`FarmRunner.write_notes()` の文面が古いままだとログ解釈と提出判断を誤る。notes template も現在の方針へ合わせる必要がある。

## 実施

- 変更対象は `experiments/neurogolf_farm/farm_runner.py` のみ。
- `write_notes()` の文面を以下に合わせた。
  - `params + output tensor bytes`
  - `Where` は MAC=0
  - `FULL_GRID_COMPOSITION` / `ScatterND` は構造riskとして別guardrail
  - `best_by_task_count`, `duplicate_task_count`, `best_total_local_delta`

## 結果

- `py_compile` 成功。
- `FarmRunner.run_smoke()` + `write_notes()` 成功。
- generated notesに `submit-gate local delta uses best-total value: 0.1` が出ることを確認。

## 判断

提出なし。notes template の tooling 較正のみで、候補bundleのlocal estimate更新ではない。

## 次アクション

farm tooling の基礎ギャップはかなり埋まった。次は実候補生成側へ戻り、`best_total_local_delta` を用いた提出可否判定で小さいscore-producing候補を探す。
