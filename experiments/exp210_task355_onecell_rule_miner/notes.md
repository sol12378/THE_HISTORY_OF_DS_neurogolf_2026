# exp210_task355_onecell_rule_miner

## 目的

exp087の1x1 cropish上位task355について、単純な入力集約ruleでoutput colorを説明できるか高速に確認する。

## 結果

- best_rule: `corner_br`
- pass: `70/267`
- output_hist: `{0: 17, 1: 24, 2: 42, 3: 29, 4: 20, 5: 17, 6: 31, 7: 34, 8: 28, 9: 25}`

## 判断

full passならtiny ONNX reducerへ進む。partialならnear-miss特徴を見て、短く伸ばせないなら別taskへpivotする。
