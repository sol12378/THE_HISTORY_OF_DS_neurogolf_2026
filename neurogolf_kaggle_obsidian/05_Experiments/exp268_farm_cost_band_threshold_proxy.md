# exp268_farm_cost_band_threshold_proxy

## 仮説

farm の `predicted_cost_band` は local estimate gate と同じ cost proxy 閾値を優先すべきである。低cost op が少数というだけで 36k bytes の候補を `250-600_plausible` と表示すると、提出判断を誤る。

## 実施

- 変更対象は `experiments/neurogolf_farm/cost_extractor.py` のみ。
- IR band 判定から `all low-cost ops and len <= 3` の override を削除。

## 結果

- `py_compile` 成功。
- dtype probe:
  - `uint8`: `cost_proxy=9000`, band `high_cost_probe_only`
  - `float32`: `cost_proxy=36000`, band `high_cost_probe_only`
  - `int64`: `cost_proxy=72000`, band `high_cost_probe_only`
- farm smoke成功。`smoke_channel_gather` / `smoke_slice_pad` は `cost_proxy=36000` で `high_cost_probe_only` になった。

## 判断

提出なし。tooling 較正のみであり、exp265 public LB `6008.90` を上回る候補 bundle の local estimate ではない。

## 次アクション

次は「Where MAC=0」「recolor_direct cost 44」など、既知較正と既存 hard reject / primitive 表現が矛盾していないかを、実測可能な既存artifact型probeで確認する。
