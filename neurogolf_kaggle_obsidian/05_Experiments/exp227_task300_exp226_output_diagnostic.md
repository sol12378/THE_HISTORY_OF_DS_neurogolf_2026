# exp227_task300_exp226_output_diagnostic

## 目的

exp226のexample0 mismatchをargmax gridで診断する。

## 結果

- argmax gridはexpected padded gridと一致。
- one-hot表現の背景channel問題が疑わしい。

## 判断

背景channelを明示的に復元する。
