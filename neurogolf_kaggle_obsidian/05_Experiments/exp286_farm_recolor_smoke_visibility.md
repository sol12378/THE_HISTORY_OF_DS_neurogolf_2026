# exp286_farm_recolor_smoke_visibility

## Plan

recolor direct/cast の cost差を farm smoke の標準 guardrail に出し、direct-first supplier の前提を毎回確認できるようにする。

## Do

`farm_runner.py` の `_smoke_programs()` に `recolor_direct_program()` / `recolor_cast_program()` を追加。

## Check

- direct: `cost_proxy=45`, `param_count=44`, `memory_bytes_proxy=1`
- cast: `cost_proxy=141`, `param_count=140`, `memory_bytes_proxy=1`
- `submission_decision_scope=smoke_dummy_not_kaggle_candidate`

## Act

提出なし。実候補bundleのlocal estimate改善ではない。
