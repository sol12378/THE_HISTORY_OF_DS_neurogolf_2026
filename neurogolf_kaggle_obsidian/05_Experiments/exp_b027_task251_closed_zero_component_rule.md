# exp_b027_task251_closed_zero_component_rule

## 目的

`exp070` でLOCAL_PREDICATE_FILL_COMPILER上位の中でも単純そうだった task251 を、説明可能な sparse/region fill rule に圧縮できるか確認する。

## 結果

- task: `251`
- full pass rule: `closed_zero_component_neighbor2_to_1`
- validation: `266/266`
- train: `3/3`
- test: `1/1`
- arc-gen: `262/262`
- avg changed cells: `10.18`
- max changed cells: `28`

## Rule

0-component がgrid borderへ接しておらず、そのcomponentの4-neighbor境界色が `{2}` だけなら、そのcomponentを color `1` にする。

## Negative Check

安いrow/column reduction近似として「上下左右のnearest nonzeroがすべて2なら1」を試したが、`195/266` に留まった。task251は単なるrow/col enclosureではなく、真のzero-component connectivityが必要。

## Lowering Implication

素朴なflood-fill unrollは過去実験で高cost化しているため、そのままONNX化しない。次は border reachability、rectangle-specific closed mask、または小kernel iterative maskをstrict cost gate付きで比較する。

## Decision

ruleは採用候補。ONNX lowering未実装のため提出なし。次は `exp_b028` で cost-gated closed-component lowering を試す。
