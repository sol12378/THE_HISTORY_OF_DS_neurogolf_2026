# exp283_farm_recolor_primitives

## 仮説

`recolor_direct` と `recolor_cast` は cost が大きく違うため、IR primitive としても区別できるべきである。これにより候補生成とrankingで direct lane を優先しやすくなる。

## 実施

- 変更対象は `experiments/neurogolf_farm/ir.py` のみ。
- `PrimitiveKind.RECOLOR_DIRECT` と `PrimitiveKind.RECOLOR_CAST` を追加。

## 結果

- direct: primitive kind `recolor_direct`, `cost_proxy=45`
- cast: primitive kind `recolor_cast`, `cost_proxy=141`
- farm smoke成功。

## 判断

提出なし。IR 表現追加のみで、候補bundleのlocal estimate更新ではない。

## 次アクション

実候補生成へ戻り、recolor supplier が direct/cast primitive を明示して `submission_decision` まで出す。
