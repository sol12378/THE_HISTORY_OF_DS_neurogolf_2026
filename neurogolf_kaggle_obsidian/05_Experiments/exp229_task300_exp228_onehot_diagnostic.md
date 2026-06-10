# exp229_task300_exp228_onehot_diagnostic

## 目的

exp228のmismatchをone-hot tensor差分で診断する。

## 結果

- top-left 4x3内のchannel sumsは一致。
- full outputではchannel0が `892` vs expected `4`。
- padding外側に888個余分なchannel0が立っていた。

## 判断

padding外側は全channel zeroに戻す。
