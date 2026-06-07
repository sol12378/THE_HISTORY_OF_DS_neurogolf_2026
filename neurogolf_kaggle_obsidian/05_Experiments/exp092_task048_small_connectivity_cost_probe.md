# exp092_task048_small_connectivity_cost_probe

## 目的

`task048` の8x8小領域connectivity ruleをONNXへ落とす余地があるか、bounded dilation proxyでcostを測る。

## 結果

- `onecell_output_floor`: cost `52`
- `reachability_4step_8x8`: cost `7214`
- `reachability_8step_8x8`: cost `10542`

## 解釈

出力1x1のfloorは非常に軽いが、connectivity unrollは8x8でも中間tensorとConv/Greater/Cast/Mulが積み上がり、600級には届かない。

## Decision

task048はrule hitとして保持するが、naive connectivity unrollは不採用。次は既存artifact surgery、または閉形式のpath特徴を探す。

## Risk

- leakage risk: low。
- overfitting risk: medium。proxyであり提出候補ではない。
