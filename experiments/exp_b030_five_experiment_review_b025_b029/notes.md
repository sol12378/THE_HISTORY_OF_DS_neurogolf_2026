# exp_b030_five_experiment_review_b025_b029

## 目的

5実験レビュー。`exp_b025`〜`exp_b029` が、LB最大化・cost<=250・7700方針に本当に有意義だったか点検する。

## Verdict

`useful_but_lowering_gap_remains`

## 有意義だった点

- `exp_b025` でcurrent submit-safe bestをLB `5930.40` へ更新した。
- `exp_b026` でpost-pass偏重を認識し、rule/compiler本体へ戻る判断をした。
- `exp_b027` でtask251をclosed zero-component ruleとして `266/266` 解けた。
- `exp_b028` でnaive flood-fill unrollがtask251でも高costと確認した。
- `exp_b029` で既存artifactのreachability depth pruningも不成立と確認した。

## 足りない点

- task251はrule hitだが、cost改善ONNXがない。
- cost<=250達成task数は増えていない。
- region-fill laneは正解ruleよりloweringがボトルネック。

## 次PDCA

- task251で続けるならrectangle-specific closed maskだけに絞る。
- それがすぐ安くならない場合、L3 object move / object-anchor crop / low-cost artifact profileへ戻る。
- 提出は引き続きfull-arc-safe positive deltaだけに限定する。
