# exp285_farm_recolor_ir_factories

## 目的

`recolor_direct` / `recolor_cast` を enum で区別するだけでなく、supplier が使う標準 IR factory を `ir.py` に追加する。

## 仮説

recolor 系候補は `direct` と `cast fallback` の cost 差が大きい。factory を用意すると、候補生成側が direct-first の方針を明示でき、cast-heavy 候補を誤って同列に扱うリスクを下げられる。

## 実装

- `recolor_direct_program()`
  - 1 node `Gather`
  - `PrimitiveKind.RECOLOR_DIRECT`
  - default `param_count=44`
- `recolor_cast_program()`
  - `Gather` + `Cast`
  - `PrimitiveKind.RECOLOR_CAST`
  - default `param_count=44+96`

## 結果

```json
{
  "direct_cost_proxy": 45,
  "cast_cost_proxy": 141,
  "direct_band": "250-600_plausible",
  "cast_band": "250-600_plausible"
}
```

## 判定

成功。supplier は `recolor_direct_program()` を優先し、`recolor_cast_program()` は明示的な fallback として扱う。

## Submit

なし。IR tooling のみで、実候補bundleのlocal estimate改善ではない。

## Leakage / Overfitting risk

- Leakage risk: なし
- Overfitting risk: なし
