# exp083_sparse_update_primitive_cost_probe

## 目的

`task366` のsmall-patch / sparse-coordinate lowering候補として、少数 `ScatterND` 更新や小さい `Gather` の公式costを測る。

## 結果

- scored variants: `11/11`
- min scored cost: `10418`

## 判断

これは提出候補ではない。小さいsparse updateでも数千costなら、task366を600級へ落とすには別表現が必要。

## Risk

- leakage risk: low。合成primitiveのみ。
- overfitting risk: low。cost-only診断。
