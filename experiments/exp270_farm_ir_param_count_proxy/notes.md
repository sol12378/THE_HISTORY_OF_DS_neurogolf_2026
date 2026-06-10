# exp270_farm_ir_param_count_proxy

## 目的

既知較正の `recolor_direct(cost 44)` と `recolor_cast(cost 140)` の差を farm local estimate に載せる。現状の IR path は `param_count=0` 固定だったため、output bytes 以外の小さい param 差を表現できなかった。

## 変更

- `experiments/neurogolf_farm/cost_extractor.py` で、各 `IRNode.attrs["param_count"]` を `param_count_proxy` に加算するようにした。
- `cost_proxy = param_count_proxy + memory_bytes_proxy` に変更した。
- ONNX path は既に initializer の `param_count` を使っているため変更していない。

## 確認

- `py_compile` 成功。
- recolor param probe:
  - `recolor_direct`: `param_count=44`, output bytes `1`, `cost_proxy=45`
  - `recolor_cast`: `param_count=140`, output bytes `1`, `cost_proxy=141`
- `FarmRunner.run_smoke()` 成功。

## 判断

提出なし。これは tooling 較正であり、exp265 public LB `6008.90` を上回る候補 bundle の local estimate 更新ではない。

## リスクと次アクション

- leakage risk: なし。
- overfitting risk: なし。
- 次は farm の smoke/IR supplier 側で `recolor_direct` を優先し、`recolor_cast` を同等候補として扱わないように candidate ranking へ接続する。
