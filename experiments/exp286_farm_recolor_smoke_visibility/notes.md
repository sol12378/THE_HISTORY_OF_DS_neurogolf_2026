# exp286_farm_recolor_smoke_visibility

## Plan

Phase C の farm supplier 復帰に向け、`recolor_direct` / `recolor_cast` の cost差を farm smoke の標準 guardrail に常時表示する。

見込み効果:

- `recolor_direct`: params 44 + output bytes 1 = `cost_proxy=45`
- `recolor_cast`: params 140 + output bytes 1 = `cost_proxy=141`

この差を smoke に出すことで、supplier 実装時に direct-first の ranking が崩れていないかを毎回確認できる。

## Do

`experiments/neurogolf_farm/farm_runner.py` の `_smoke_programs()` に、exp285 の `recolor_direct_program()` / `recolor_cast_program()` を追加した。

## Check

farm smoke:

```json
{
  "direct": {
    "predicted_cost_band": "250-600_plausible",
    "param_count": 44,
    "memory_bytes_proxy": 1,
    "cost_proxy": 45
  },
  "cast": {
    "predicted_cost_band": "250-600_plausible",
    "param_count": 140,
    "memory_bytes_proxy": 1,
    "cost_proxy": 141
  }
}
```

Accuracy gate:

- smoke-only IR guardrail なので task accuracy は対象外。
- `submission_decision_scope=smoke_dummy_not_kaggle_candidate` を確認。

## Act

No submit。実候補bundleの local estimate 改善ではなく、farm visibility のみ。

## Risk

- Leakage risk: なし
- Overfitting risk: なし
- Operational risk: smoke dummy の `should_submit=true` を実提出判断と混同しない。
