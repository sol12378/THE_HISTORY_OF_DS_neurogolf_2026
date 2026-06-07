# exp_b013_five_experiment_review_b007_b011

## 目的

exp_b007-b011の5実験が、`cost<=250` / `7700` / LB最大化に本当に有意義だったかを問い直す。

## 判定

`meaningful_but_not_score_producing`

## 有意義だった点

- b007-b009でcoordinate-first sparse fill grammarを棄却できた。
- b010でtask037の説明可能rule `opposite_ray_diag_same_color` が `266/266` full passした。
- b011でfull-grid Conv visibility loweringが正しくても高costになることを実測した。

## 不十分だった点

- score deltaは `0.0`。
- task037 ruleはまだ `cost<=250` に落ちていない。
- P0 sparse fill全体への横展開は未完。

## Decision

次PDCAはrule探索よりもlow-cost lowering専用へ寄せる。task037はdiagonal-specific small mask / triangular Gather / graph surgeryで<=600、理想<=250を狙う。b010 rule bankはpartial evidenceを使ってdecision tree合成へ拡張する。
