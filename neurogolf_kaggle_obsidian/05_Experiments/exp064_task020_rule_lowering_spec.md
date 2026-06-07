# exp064_task020_rule_lowering_spec

## Hypothesis

task020の明示ruleをONNX化するには、dynamic bbox込みのlowering blockを先に固定し、correctness-firstとcost-optimizedを分ける必要がある。

## Result

Lowering blocks:

1. `bbox5_crop`: nonzero bboxを検出し5x5 local frameへ揃える。
2. `target_color_roles`: target colorごとにcorners / edge_mid / inner / centerのgroup countを計算する。
3. `class_selector`: exp062 ruleをEqual/Greater/Whereで実装する。
4. `orientation_and_missing`: classごとのD4 variantから互換variantを選び、missing 3 cellsを決める。
5. `writeback`: 5x5 local fill maskを30x30 one-hot outputへ戻す。

## Acceptance

- first target: `266/266` all arc-gen pass and cost `< 90133`
- stretch target: cost `<=600`
- submit policy: official validation passかつstrict seedより改善した場合だけKaggle single-task delta提出

## Decision

次はcorrectness-first ONNXを生成する。raw 24-case tableやteacher artifactは使わない。
