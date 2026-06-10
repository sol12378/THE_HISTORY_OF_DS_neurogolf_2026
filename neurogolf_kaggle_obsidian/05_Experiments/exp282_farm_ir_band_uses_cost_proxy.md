# exp282_farm_ir_band_uses_cost_proxy

## 仮説

IR band判定は `params + output tensor bytes` の `cost_proxy` を見るべきである。`memory_bytes_proxy` だけを見ると、paramsが大きい候補を低cost扱いしてしまう。

## 実施

- 変更対象は `experiments/neurogolf_farm/cost_extractor.py` のみ。
- `assess_ir()` の band 判定を `memory_bytes_proxy` から `cost_proxy` へ変更。

## 結果

- `param_count=44`, output bytes `1`: `cost_proxy=45`, band `250-600_plausible`
- `param_count=5000`, output bytes `1`: `cost_proxy=5001`, band `high_cost_probe_only`
- farm smoke成功。

## 判断

提出なし。tooling band較正のみで、候補bundleのlocal estimate更新ではない。

## 次アクション

実候補生成へ戻り、cost_proxy/band/submission_decisionを使って提出可否を記録する。
