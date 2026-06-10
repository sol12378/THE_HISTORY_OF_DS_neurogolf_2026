# exp206_task185_dynamic_output_diagnostic

## 目的

exp204 dynamic candidateのexample 0 mismatchをtensor diffで診断する。

## 結果

- `dynamic_axis_basic`: channel0だけを3x3に出力。expectedはchannel0/1/3混在。
- `dynamic_axis_nonzero_only`: 出力が全zero。
- `dynamic_axis_score_nonzero_core_basic`: channel1を2セル出すが、expected位置とズレる。
- `dynamic_axis_score_nonzero_default_bg`: 背景補完は入るが、channel1位置ズレとchannel3欠落が残る。

## 判断

背景補完だけでは不十分。selector indexから4x4 latticeへの対応、または3x3 output位置への対応がズレている可能性が高い。次は小さなdebug graphでselected index/selected latticeを直接確認する。

## リスク

- leakage risk: low。validation exampleのdebug比較。
- overfitting risk: medium。example 0診断のみなので、修正後はfull validation必須。
