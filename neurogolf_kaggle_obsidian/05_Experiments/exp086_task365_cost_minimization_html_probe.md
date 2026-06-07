# exp086_task365_cost_minimization_html_probe

## 目的

ユーザー提供HTMLのONNX cost最小化情報を基に、`task365` dense rectangle crop が `cost<=250〜600` 級へ落とせるかを公式 `score_network` で小proxy実測する。

## 仮説

`Slice/Gather/Pad` のFREE-op中心で小領域だけを扱えば、`task365` の `max_count2` object crop rule は既存artifact cost `58814` から大きく削れる。ただし公式検証が padded one-hot tensor を直接比較するため、Padなし小出力は提出候補として使えない可能性がある。

## 結果

- `slice_3x3_small_output`: cost `12`
- `slice_6x6_small_output`: cost `12`
- `slice_3x3_pad30`: cost `381`
- `slice_6x6_pad30`: cost `1461`
- `gather_6x6_pad30`: cost `8661`
- `small_equal_6x6_pad30`: cost `3262`

## 解釈

Padなし小出力は `score_network` では測れるが、公式 `verify_subset` は `run_network` 出力と `convert_to_numpy(example)["output"]` を `np.array_equal` で直接比較する。したがって、Padなし `1x10x3x3` / `1x10x6x6` はcost診断としては有効でも、`task365` の正答候補にはならない。

`task365` は selected shape に `(6,6)` が存在するため、hidden汎化も考えると少なくとも6x6相当のcrop領域を30x30 one-hot tensor上に表現する必要がある。selectorなしの `Slice+Pad` proxyだけで `1461` なので、full single-modelで `cost<=600` は厳しい。

## Decision

HTMLのFREE-op小領域化は大幅削減には有効だが、`task365` を250〜600級へ直接落とす主ルートではない。次は最大output areaが14セル以下のcrop task、または6x6 memory floorを避ける別表現を探す。

## Risk

- leakage risk: low。cost-only probeでoutput lookupはない。
- overfitting risk: low。提出候補ではない。
