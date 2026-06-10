# exp268_farm_cost_band_threshold_proxy

## 目的

exp267で output tensor bytes の cost proxy を正した後も、低cost op が少数なら `cost_proxy=36000` でも `250-600_plausible` と判定される古い band override が残っていた。提出前 local estimate gate と矛盾するため、band 判定を `cost_proxy` 優先へ揃える。

## 変更

- `experiments/neurogolf_farm/cost_extractor.py` の IR band 判定から、`all low-cost ops and len <= 3` だけで `250-600_plausible` にする分岐を削除した。
- output bytes 計算や ONNX path は変更していない。

## 確認

- `py_compile` 成功。
- dtype band probe:
  - `uint8`: `cost_proxy=9000`, `predicted_cost_band=high_cost_probe_only`
  - `float32`: `cost_proxy=36000`, `predicted_cost_band=high_cost_probe_only`
  - `int64`: `cost_proxy=72000`, `predicted_cost_band=high_cost_probe_only`
- `FarmRunner.run_smoke()` 成功。`smoke_channel_gather` と `smoke_slice_pad` は `cost_proxy=36000` / `high_cost_probe_only` になり、exp267の矛盾が解消した。

## 判断

提出なし。これは tooling band 較正であり、exp265 public LB `6008.90` を上回る候補 bundle の local estimate 更新ではない。

## リスクと次アクション

- leakage risk: なし。
- overfitting risk: なし。
- 次は `Where` / `ScatterND` の hard reject が「Where MAC=0」の較正と矛盾するかを、公式 cost 実測可能な既存artifact型probeで確認する。
