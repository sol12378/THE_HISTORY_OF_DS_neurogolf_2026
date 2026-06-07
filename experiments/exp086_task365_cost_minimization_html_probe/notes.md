# exp086_task365_cost_minimization_html_probe

## 目的

ユーザー提供HTMLのONNX cost最小化技法が、`task365` dense rectangle cropを `cost<=250〜600` 級へ落とす現実的余地を持つか、公式 `score_network` で小さなproxyを実測する。

## 仮説

`Slice/Gather/Pad` だけで小cropを扱えれば、`task365` の正答rule (`max_count2` object crop) は既存artifact cost `58814` から大きく削れる。ただし公式出力が `[1,10,30,30]` 固定の場合、Pad後のmemory floorだけで600を超える可能性がある。

## 結果

- scored variants: `7/7`
- min scored cost: `0`
- baseline cost: `58814`

## 判断

score_network alone accepts small static outputs at cost 12, but official validation compares padded one-hot tensors directly. For a full task365 solution the output must behave as [1,10,30,30]; the minimal 6x6 Slice+Pad proxy already costs 1461 before object selection. Therefore HTML-style FREE-op minimization is useful for large reductions, but task365 is unlikely to reach cost<=600 unless the rule can avoid the 6x6 crop memory floor or split into a smaller shape-specific artifact, which a single task model cannot generally do.

## 追加解釈

`score_network` は小さいgraph outputも計測できるが、公式 `verify_subset` は `convert_to_numpy(example)["output"]` と `run_network` 出力を `np.array_equal` で直接比較する。したがって、Padなし `1x10x3x3` / `1x10x6x6` はcost診断としては有効でも、`task365` の提出候補にはならない。

`task365` は selected shape に `(6,6)` が1例あり、hidden側でも最大shapeが出る可能性を考えると、少なくとも `6x6` 相当のcrop領域をpadded one-hot上に表現する必要がある。`Slice+Pad` proxyの `1461` が、selectorなしの下限に近い。

## Risk

- leakage risk: low。cost-only probeで、output lookupはない。
- overfitting risk: low。提出候補ではない。
