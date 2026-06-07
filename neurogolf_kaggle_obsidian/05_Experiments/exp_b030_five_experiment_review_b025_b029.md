# exp_b030_five_experiment_review_b025_b029

## 目的

`exp_b025`〜`exp_b029` の5実験が、LB最大化・cost<=250・7700方針に本当に有意義だったか点検する。

## Verdict

`useful_but_lowering_gap_remains`

## 評価

- `exp_b025`: 有意義。submit-safe bestをLB `5930.40` へ更新。
- `exp_b026`: 有意義。post-pass偏重を認識し、rule/compiler本体へ戻る判断を明文化。
- `exp_b027`: 有意義。task251を `266/266` の説明可能ruleへ圧縮。
- `exp_b028`: 有意義な負の結果。naive flood-fill unrollはfull passでもcost負け。
- `exp_b029`: 有意義な負の結果。既存artifactのreachability depth pruningは不成立。

## 結論

この5実験は目標に対して有意義だった。LBは更新し、rule/compiler laneにも戻れた。ただしcost<=250の実装は進んでいない。task251は「ruleは解けたがloweringが未解決」の代表例として扱う。

## 次の方針

- task251継続ならrectangle-specific closed maskに限定する。
- すぐ安くならない場合、L3 object move / object-anchor crop / low-cost artifact profileへ移る。
- submitは引き続きfull-arc-safe positive deltaのみ。
