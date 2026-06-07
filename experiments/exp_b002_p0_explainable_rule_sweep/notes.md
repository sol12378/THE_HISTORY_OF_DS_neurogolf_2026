# exp_b002_p0_explainable_rule_sweep

## 目的

exp_b routeの最初のrule miningとして、P0上位40 taskに対して、説明可能かつ安いlowering候補になりうるrule familyを横断探索する。

## 仮説

signature lookup teacherの一部は、D4 orbit completion、rectangle closure、row/column segment fillのような小さな幾何ruleで置換できる。

## 結果

- scanned tasks: 40
- evaluated rows: 120
- full pass hits: 0
- train/test pass hits: 0
- best partial: {'task_id': 286, 'rank': 1, 'family': 'signature_lookup_current', 'rule_name': 'color_bbox_rectangle_closure', 'status': 'partial', 'total_pass': 0, 'total_examples': 265, 'train_pass': 0, 'train_examples': 2, 'test_pass': 0, 'test_examples': 1, 'arc_pass': 0, 'arc_examples': 262, 'fail_reasons': '{"candidate_count=0": 265}', 'lowering_plan': 'color Equal mask, Reduce row/column bbox if cheap, then static rectangle mask', 'reject_risk': 'medium: bbox detection must avoid NonZero/Compress/full-grid GatherND'}

## 解釈

full pass hitがあれば次にONNX loweringへ進む。full passがなければ、失敗理由を見てrule familyを追加する。local/LB較正のため、採用はsmall delta submission単位に限定する。

## Risk

- leakage risk: 低。arc-gen label lookupやteacher output tableは使っていない。
- overfitting risk: 中。P0 taskに対する固定rule family sweepなので、hit後はfamily holdoutとKaggle small submissionが必要。
