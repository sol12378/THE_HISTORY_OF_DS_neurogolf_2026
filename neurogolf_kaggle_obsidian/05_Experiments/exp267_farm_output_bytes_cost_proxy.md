# exp267_farm_output_bytes_cost_proxy

## 仮説

roadmap の Phase B/C を進める前に、farm tooling の cost proxy を「params + output tensor bytes」に合わせる必要がある。現状の `cost_extractor.py` は中間 tensor を足し、最終 `output` を除外していたため、dtype memory lane の判断を誤る。

## 実施

- 変更対象は `experiments/neurogolf_farm/cost_extractor.py` のみ。
- IR path: `node.output.name == "output"` の tensor bytes だけを `memory_bytes_proxy` に加算。
- ONNX path: `graph.output` の tensor bytes を `memory_bytes_proxy` に採用。
- 中間 full-grid guard と `Where` / `ScatterND` hard reject は別施策として保持。

## 結果

- `py_compile` 成功。
- dtype probe:
  - `uint8`: `cost_proxy=9000`
  - `float32`: `cost_proxy=36000`
  - `int64`: `cost_proxy=72000`
- `FarmRunner.run_smoke()` 成功。

## 判断

提出なし。これは tooling 較正であり、exp265 public LB `6008.90` を上回る新しい候補 bundle の local estimate ではない。

## 次アクション

`cost_proxy=36000` でも `predicted_cost_band=250-600_plausible` と出る古い band 判定が残っている。次の1施策では band 判定を `cost_proxy` 優先へ修正し、local estimate gate と提出判断の整合性を上げる。
