# exp091_task048_bridge_connectivity_rule

## 目的

`exp087` の1x1 P0候補 `task048` の説明可能ruleを確認する。

## 結果

- pass: `270/270`
- baseline cost: `5609`

## Rule

色8の4-connected componentが2つの色2 componentの両方に接していれば出力は `[[8]]`。そうでなければ `[[0]]`。

## Decision

full passなら、次は8x8小領域connectivity loweringのcost probeへ進む。出力1x1なので、task185より600級候補として強い。

## Risk

- leakage risk: low。明示的な連結性rule。
- overfitting risk: medium-low。色2/8固定だがall arc-gen pass。
