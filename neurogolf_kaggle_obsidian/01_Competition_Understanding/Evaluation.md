# Evaluation

Document the official metric, local validation proxy, and cases where local scores disagree with leaderboard results.

## 2026-06-05 公式確認

- taskごとの得点は、functional correctなnetworkに対して `max(1, 25 - ln(cost))`。
- `cost = parameter count + memory footprint bytes`。
- 2026-05-04 updateでMACsはobjectiveから除外された。今後の最小化対象はmemoryとparams。
- Functional correctnessは公開ARC-AGI例とprivate benchmark suiteで検証される。全テストで完全一致しないtaskは得点対象にならない。
- 公式utilityでは `onnxruntime` profiler traceからmemoryを計測し、`calculate_params`でinitializer/Constant等を数える。
- exp001/task087 local: ARC-AGI `5 pass / 0 fail`、ARC-GEN `261 pass / 0 fail`、memory `36000`、params `60`、推定 `14.507` points。

## ローカル検証proxy

- 配布 `neurogolf_utils.py` の `convert_to_numpy`, `run_network`, `score_network` 相当を使う。
- graph optimizationは無効化し、profilingを有効化する。
- static shape、単一input/output、禁止opなしを事前確認する。
