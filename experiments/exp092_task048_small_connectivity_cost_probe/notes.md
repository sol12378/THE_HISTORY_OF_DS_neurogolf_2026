# exp092_task048_small_connectivity_cost_probe

## 目的

`task048` の8x8小領域connectivity ruleをONNXへ落とす余地があるか、bounded dilation proxyでcostを測る。

## 結果

`result.json` / `cost_probe.csv` を参照。

## Decision

8-step proxyが600を大きく超える場合、naive connectivity unrollは避け、既存artifact surgeryまたは閉形式のpath特徴を探す。

## Risk

- leakage risk: low。
- overfitting risk: medium。proxyであり提出候補ではない。
