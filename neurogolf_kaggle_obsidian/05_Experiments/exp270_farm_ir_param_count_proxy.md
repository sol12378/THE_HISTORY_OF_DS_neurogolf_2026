# exp270_farm_ir_param_count_proxy

## 仮説

farm の IR local estimate が `params + output tensor bytes` である以上、IR node 側でも `params` を明示できる必要がある。これがないと `recolor_direct(cost 44)` と `recolor_cast(cost 140)` の差が candidate selection に反映されない。

## 実施

- 変更対象は `experiments/neurogolf_farm/cost_extractor.py` のみ。
- `IRNode.attrs["param_count"]` を `param_count_proxy` として加算。
- `cost_proxy = param_count_proxy + memory_bytes_proxy` に変更。

## 結果

- `recolor_direct`: `param_count=44`, output bytes `1`, `cost_proxy=45`
- `recolor_cast`: `param_count=140`, output bytes `1`, `cost_proxy=141`
- farm smoke成功。

## 判断

提出なし。tooling param proxy 較正のみで、候補bundleのlocal estimate更新ではない。

## 次アクション

候補生成・ranking で `recolor_direct` を `recolor_cast` より優先できるよう、farm supplier / ledger 側の表現に接続する。
