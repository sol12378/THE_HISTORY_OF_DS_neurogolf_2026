# exp283_farm_recolor_primitives

## 目的

既知較正では `recolor_direct(cost 44)` が `recolor_cast(cost 140)` より安い。exp270で `attrs["param_count"]` によるcost差は表現できるようになったが、IR primitive kind として direct/cast を区別できなかった。候補生成・ranking側で明示的に扱えるよう、IR enum に足場を追加する。

## 変更

- `experiments/neurogolf_farm/ir.py` の `PrimitiveKind` に以下を追加した。
  - `RECOLOR_DIRECT = "recolor_direct"`
  - `RECOLOR_CAST = "recolor_cast"`

## 確認

- `py_compile` 成功。
- IR/cost probe:
  - `PrimitiveKind.RECOLOR_DIRECT`, `param_count=44`, output 1 byte -> `cost_proxy=45`, primitive kind `recolor_direct`
  - `PrimitiveKind.RECOLOR_CAST`, `param_count=140`, output 1 byte -> `cost_proxy=141`, primitive kind `recolor_cast`
- `FarmRunner.run_smoke()` 成功。

## 判断

提出なし。これは IR 表現の tooling 較正であり、実候補bundleのlocal estimate更新ではない。

## リスクと次アクション

- leakage risk: なし。
- overfitting risk: なし。
- 次はこの primitive を使う実候補 supplier / rewrite へ戻る。
