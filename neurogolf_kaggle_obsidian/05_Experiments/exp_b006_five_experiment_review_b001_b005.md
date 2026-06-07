# exp_b006_five_experiment_review_b001_b005

## 目的

exp_b001-b005が、LB最大化、全task cost<=250、7700到達に本当に有意義だったかを問い直す。

## 結論

この5実験はscore-producingではないが、course-correctionとして有意義だった。

- exp_b001: 400-task backlogとKaggle較正baselineを作った。
- exp_b002: queue-top P0という誤った対象設定を露出した。
- exp_b003: teacher-gain P0へ対象を修正した。
- exp_b004: 次grammarがbbox-local/object-roleであると分かった。
- exp_b005: raw neighbor-countだけでは不足と確認した。

## 次PDCA

1. exp_b007でbbox-local/object-role feature minerを作る。
2. 対象は task020/126/251/90/37 を優先する。
3. full pass後のみ tiny mask / tiny ScatterND lowering を作る。
4. accepted candidateはcost<=250を基準にし、single-task delta submissionでLB較正する。
