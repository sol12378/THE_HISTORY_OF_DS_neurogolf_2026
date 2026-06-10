# exp313_farm_notes_next_low_cost

## Plan

Phase C の次手を notes から追えるように、farm notes に `next_low_cost_primitive` と根拠 cost を出す。見込みコスト削減は直接なし。`grid_sample` supplier 接続の PDCA handoff を明確にする。

## Do

`farm_runner.py` の `write_notes()` に以下を追加。

- `next low-cost primitive`
- `next low-cost primitive cost`
- `next low-cost subjects`

## Check

- next_low_cost_primitive: `grid_sample`
- next_low_cost_primitive_cost: `9`
- next_low_cost_subjects: `["smoke_gridsample"]`
- notes_has_grid_sample: `true`
- notes_has_cost: `true`

## Act

notes visibility のみで実候補bundleのlocal estimate改善ではないため提出なし。次は `grid_sample` supplier を実task候補へ接続する。

## Risk

- leakage risk: low。提出候補なし。
- overfitting risk: low。notes visibility の追加のみ。
