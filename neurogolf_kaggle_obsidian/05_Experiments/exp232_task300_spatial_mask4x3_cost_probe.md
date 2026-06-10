# exp232_task300_spatial_mask4x3_cost_probe

## 目的

GatherElements対象を10chから1ch spatial maskに縮小し、costを下げる。

## 結果

- validation: `267_pass_0_fail`
- cost: `81541`
- baseline: `77546`

## 判断

かなり近いがno gain。selected mask生成をさらに軽くする。
