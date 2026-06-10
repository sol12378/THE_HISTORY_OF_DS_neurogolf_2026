# exp267_farm_output_bytes_cost_proxy

## 目的

farm tooling の cost proxy を、2026-06-10 時点の較正である「params + output tensor bytes」に合わせる。今回は LB を直接上げる候補生成ではなく、次の Phase B/C 候補を誤って捨てたり誤評価したりしないための計測基盤修正である。

## 変更

- `experiments/neurogolf_farm/cost_extractor.py` の IR cost proxy を、最終 `output` tensor の byte 数だけ加算する形に修正した。
- ONNX path も `graph.value_info` の中間 tensor byte ではなく `graph.output` の byte を `memory_bytes_proxy` に使うよう修正した。
- 中間 full-grid 数の guard は残した。`Where` / `ScatterND` などの hard reject は今回は触らない。

## 確認

- `py_compile` 成功。
- dtype probe:
  - `uint8`: output 9000 elements -> `cost_proxy=9000`
  - `float32`: output 9000 elements -> `cost_proxy=36000`
  - `int64`: output 9000 elements -> `cost_proxy=72000`
- `FarmRunner.run_smoke()` 成功し、`experiments/exp267_farm_output_bytes_cost_proxy/result.json` を生成した。

## 判断

提出なし。これは tooling 較正であり、exp265 public LB `6008.90` を上回る候補 bundle の local estimate 更新ではない。

## リスクと次アクション

- leakage risk: なし。候補や生データには触れていない。
- overfitting risk: なし。tooling の cost proxy 修正のみ。
- 残ギャップ: `cost_proxy=36000` でも低cost op 少数なら `predicted_cost_band=250-600_plausible` になる古い band 判定が残っている。次の1施策で band threshold を cost proxy 優先に直す。
