# exp297_exp262_skip_task048_336_fresh_candidates

## 目的

e297_short_skip048_336 の完走結果から、exp263 partialで提出済みの17件と失敗quarantine候補 task048/task336 を除外し、未提出の task060/task226/task123 だけを exp265 current best bundle に replay して bundle化する。

## 結果

- selected_count: `3`
- failed_count: `0`
- selected_tasks: `[60, 226, 123]`
- local_delta: `0.05520869941921447`
- expected_public_lb_if_calibrated: `6008.955208699419`

## 判断

現行 best Public LB `6008.90` から local estimate が更新され、full-arc replay で失敗がなければ提出対象。

## 提出

- Kaggle ref: `53535771`
- status: `COMPLETE`
- expected public LB if calibrated: `6008.955208699419`
- public LB: `6008.96`
