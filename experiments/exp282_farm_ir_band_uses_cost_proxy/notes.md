# exp282_farm_ir_band_uses_cost_proxy

## 目的

exp270で IR の `param_count` を `cost_proxy` に入れたが、IR の `predicted_cost_band` 判定はまだ `memory_bytes_proxy` だけを見ていた。これにより、outputが小さくてもparamsが大きい候補を低cost bandに誤分類するリスクがあった。

## 変更

- `experiments/neurogolf_farm/cost_extractor.py` の `assess_ir()` で `cost_proxy = param_count_proxy + memory_bytes_proxy` を明示し、band判定を `cost_proxy` 基準に変更した。
- `cost_proxy` の返却値も同じローカル変数を使うようにした。

## 確認

- `py_compile` 成功。
- probe:
  - `param_count=44`, output bytes `1` -> `cost_proxy=45`, band `250-600_plausible`
  - `param_count=5000`, output bytes `1` -> `cost_proxy=5001`, band `high_cost_probe_only`
- `FarmRunner.run_smoke()` 成功。

## 判断

提出なし。これは IR cost band の tooling 較正であり、実候補bundleのlocal estimate更新ではない。

## リスクと次アクション

- leakage risk: なし。
- overfitting risk: なし。
- 次は実候補生成に戻り、param/output bytesが正しくbandへ反映される状態でscore-producing候補を探す。
