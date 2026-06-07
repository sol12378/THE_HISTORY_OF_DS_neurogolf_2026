# exp_b006_five_experiment_review_b001_b005

## 目的

exp_b001-b005が、LB最大化、全task cost<=250、7700到達に本当に有意義だったかを問い直す。

## 評価

| exp | verdict | reason | next action |
|---|---|---|---|
| `exp_b001_rule_replacement_backlog` | useful | 400-task backlog, cost<=250 direction, and strict seed LB calibration were established. LB 5929.89 gives a safer baseline than exp041. | Keep exp005 strict seed as calibration baseline and only add small validated deltas. |
| `exp_b002_p0_explainable_rule_sweep` | partly useful but target definition was wrong | It swept queue-top signature tasks, not teacher-gain P0. The no-hit result still rejects overly simple D4/rectangle/line rules for broad signature tasks. | Do not use queue-rank P0 alone; prioritize teacher-gain P0 plus strict high-cost tasks separately. |
| `exp_b003_teacher_gain_p0_rule_sweep` | useful negative result | Corrected target set to 18 teacher-gain P0 tasks. No full hit, but task020 D4 remained the best partial and showed sparse/orbit direction. | Continue with richer object-role grammar rather than simple geometry primitives. |
| `exp_b004_teacher_gain_p0_structural_taxonomy` | highly useful | Classified P0 tasks into grammar needs. Most are sparse color-role fill or shape/object crop, which are plausible cost<=250 targets. | Use taxonomy to choose grammar modules and avoid random rule additions. |
| `exp_b005_sparse_neighborhood_fill_sweep` | useful negative result | Pure local-neighbor counts produced zero train-fit candidates, proving P0 sparse fill needs object-role/bbox-local/symmetry features. | Add bbox-local coordinate and object-role predicates; stop using raw neighbor-count-only rules. |

## 結論

The five experiments were meaningful as a course-correction block, not as score-producing work. They established a Kaggle-calibrated strict baseline, corrected the target set, and rejected three weak grammar families. For the cost<=250 objective, the next PDCA must focus on object-role plus bbox-local coordinate rules, then tiny static-mask/ScatterND lowering.

## 次のPDCA

- exp_b007: build bbox-local/object-role feature miner for sparse background fill tasks, starting with task020/126/251/90/37.
- Require candidate rules to state target color role and target coordinates; train-fit is not enough unless all arc-gen passes.
- Only after full pass, implement lowering with tiny coordinate/mask plan and submit a single-task delta bundle.
- Track every accepted candidate against cost<=250, not merely against current artifact.
