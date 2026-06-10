# exp136_private_failure_subset_inventory

## 目的

`docs/experiment_plan_2026-06-10.md` の Phase A-2 として、local/LB gap `-352.4` を説明しうる private functional failure task 群を per-task score から推定する。

## 結果

- base: `exp_b025_submit_safe_delta_union`
- local: `6282.812218`
- LB: `5930.40`
- target gap: `352.412218`
- 400 task inventory を作成。
- subset-sum 候補: `50`
- best subset は `24` task で score sum `352.41`、gap error `0.0`。
- best subset tasks:
  - `002 004 008 009 013 014 016 018 019 021 023 024 025 029 031 032 034 036 046 047 049 050 051 366`

## 解釈

約24 task の private failure で gap が説明できるという計画書の仮説と強く整合した。ただし subset-sum は直接証拠ではなく、候補集合は複数存在する。次は上位候補集合の consensus task と provenance/shape 外挿リスクを監査し、修復または bisection probe の対象を絞る。

## 成果物

- `experiments/exp136_private_failure_subset_inventory/task_risk_inventory.csv`
- `experiments/exp136_private_failure_subset_inventory/subset_candidates.csv`
- `experiments/exp136_private_failure_subset_inventory/source_gap_options.csv`
- `experiments/exp136_private_failure_subset_inventory/result.json`

## リスク

- leakage risk: low。診断のみで提出なし。
- overfitting risk: medium。subset-sum は状況証拠であり、private failure task の証明には追加監査または probe が必要。
