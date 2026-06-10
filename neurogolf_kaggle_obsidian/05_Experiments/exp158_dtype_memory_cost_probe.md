# exp158_dtype_memory_cost_probe

## 目的

Phase C-1として、full-grid中間tensorのdtypeを `fp16` / `bool` / `uint8` へ縮小した場合に、公式 `score_network` のmemory costがdtype幅に比例して下がるかを測る。

## 結果

| case | memory | params | cost |
|---|---:|---:|---:|
| identity | 0 | 0 | 0 |
| fp32_add_zero | 0 | 1 | 1 |
| fp32_add_zero_then_identity | 36000 | 1 | 36001 |
| fp32_mul_one_then_add_zero | 36000 | 2 | 36002 |
| fp16_roundtrip | 18000 | 0 | 18000 |
| bool_roundtrip | 9000 | 0 | 9000 |
| uint8_roundtrip | 9000 | 0 | 9000 |
| int32_roundtrip | 36000 | 0 | 36000 |
| fp16_add_zero_roundtrip | 36000 | 1 | 36001 |
| bool_where_roundtrip | 9000 | 9000 | 18000 |

## 解釈

公式costはdtype幅を反映している。full-grid fp32中間は `36000`、fp16は `18000`、bool/uint8は `9000`。一方、演算を追加すると中間数が増えるため、`fp16_add_zero_roundtrip` は `36001` まで戻る。単純な全graph fp16化ではなく、full-grid mask / cast / boolean laneを狙うtargeted post-passが有望。

## 判断

Phase C-1は継続価値あり。次は既存artifactから高cost full-grid boolean/mask系中間を持つtaskを選び、`bool`/`uint8` 中間化または不要fp32 materialization削減のpost-pass候補を小さく試す。

## リスク

- leakage risk: low。合成cost probeであり、task labelは使っていない。
- overfitting risk: low。public LBではなく公式cost物理の測定。
