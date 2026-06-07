# exp_b008_shape_conditioned_sparse_fill

## 目的

固定relative座標ではなく、bbox shape / color signatureで分岐するrelative fill ruleを試す。cost<=250にはbranch数が小さいものだけが候補。

## 結果

- target tasks: 9
- evaluated candidates: 21
- full pass hits: 0
- train/test pass hits: 0
- best partial: {'task_id': 126, 'candidate': 'bbox_shape:const_4:{"(2, 3)": [[4, 1]], "(3, 6)": [[3, 1], [3, 4]], "(4, 7)": [[7, 1], [7, 5]]}', 'status': 'partial', 'train_pass': 3, 'train_examples': 3, 'test_pass': 0, 'test_examples': 1, 'arc_pass': 11, 'arc_examples': 262, 'total_pass': 14, 'total_examples': 266, 'branch_count': 3, 'avg_cells_per_branch': 1.6666666666666667, 'estimated_lowering_cost_class': 'target_cost<=250_possible', 'fail_reasons': '{"unseen_key": 192, "ok": 31, "cell_oob": 29}', 'lowering_plan': 'small decision tree over bbox/shape key plus tiny coordinate fill; submit only if full pass and branch table remains tiny'}

## 判断

full passかつbranch tableが小さいものだけloweringへ進む。unseen_keyが多ければ、shape keyではなくobject-role/symmetry生成規則が必要。
