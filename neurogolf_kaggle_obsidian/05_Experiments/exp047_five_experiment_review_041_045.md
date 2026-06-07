# exp047_five_experiment_review_041_045

## 目的

5実験ごとのPDCA品質レビュー。

## 結論

直近5実験は mixed but useful。スコア改善は出ていないが、7700へ向かううえで重要な棄却と方針転換が得られた。

## 有意義だった点

- exp041提出LB 3417.71で、lookup/artifact surgery系local upperがsubmit-safeでないと確定した。
- exp042でDSL/DAG探索に大きなproxy upsideがあると見えた。
- exp043/044でnaive loweringとfull-grid dynamic indexを棄却できた。
- exp045で7700をall400 cost<=250のfamily-level targetとして再定義した。

## 不十分だった点

- exp041のsurgery深掘りは目標へ直接効いていない。
- exp043は小さい固定変換に寄りすぎた。
- exp044は正しいruleでもloweringが悪ければ破綻する再確認だった。

## 次の方針

submit-safe seed inventory、lookup compression、family holdout、object/crop compiler、line/region rule searcherに集中する。
