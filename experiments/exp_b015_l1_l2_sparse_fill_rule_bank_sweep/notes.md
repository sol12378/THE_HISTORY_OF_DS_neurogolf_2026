# exp_b015_l1_l2_sparse_fill_rule_bank_sweep

## 目的

exp_b010でtask037 full passを出したcomponent/object-role sparse fill rule bankを、signature lookup taxonomyのL1/L2 lane全体へ横展開する。

## 結果

- target tasks: 20
- rule count: 13
- evaluated candidates: 260
- full pass hits: 0
- train/test hits: 0
- train-fit hits: 0
- best candidates: [{'task_id': 20, 'compiler_lane': 'L1_static_sparse_background_fill', 'strict_cost': 90133, 'gain_to_600': 5.012111981034998, 'gain_to_250': 5.8875807183889, 'candidate': 'opposite_ray_diag_same_color', 'status': 'partial', 'train_pass': 0, 'train_examples': 3, 'test_pass': 0, 'test_examples': 1, 'arc_pass': 14, 'arc_examples': 262, 'total_pass': 14, 'total_examples': 266, 'pass_rate': 0.05263157894736842, 'avg_pred_changed_cells': 0.9135338345864662, 'fail_reasons': '{"ok": 252}', 'lowering_plan': 'diagonal reductions or small directional scans'}, {'task_id': 133, 'compiler_lane': 'L2_local_predicate_sparse_fill', 'strict_cost': 193923, 'gain_to_600': 5.778286796824304, 'gain_to_250': 6.653755534178206, 'candidate': 'zero_holes_majority_border_color', 'status': 'partial', 'train_pass': 0, 'train_examples': 4, 'test_pass': 0, 'test_examples': 1, 'arc_pass': 0, 'arc_examples': 262, 'total_pass': 0, 'total_examples': 267, 'pass_rate': 0.0, 'avg_pred_changed_cells': 0.0, 'fail_reasons': '{"ok": 267}', 'lowering_plan': 'component hole mask plus neighboring color vote'}, {'task_id': 173, 'compiler_lane': 'L2_local_predicate_sparse_fill', 'strict_cost': 175243, 'gain_to_600': 5.6769992059442895, 'gain_to_250': 6.552467943298192, 'candidate': 'component_bbox_area16', 'status': 'partial', 'train_pass': 0, 'train_examples': 3, 'test_pass': 0, 'test_examples': 1, 'arc_pass': 0, 'arc_examples': 262, 'total_pass': 0, 'total_examples': 266, 'pass_rate': 0.0, 'avg_pred_changed_cells': 0.0, 'fail_reasons': '{"ok": 266}', 'lowering_plan': 'small component bbox masks only'}, {'task_id': 286, 'compiler_lane': 'L2_local_predicate_sparse_fill', 'strict_cost': 162199, 'gain_to_600': 5.599649600200182, 'gain_to_250': 6.475118337554084, 'candidate': 'component_bbox_area16', 'status': 'partial', 'train_pass': 0, 'train_examples': 2, 'test_pass': 0, 'test_examples': 1, 'arc_pass': 0, 'arc_examples': 262, 'total_pass': 0, 'total_examples': 265, 'pass_rate': 0.0, 'avg_pred_changed_cells': 2.4037735849056605, 'fail_reasons': '{"ok": 265}', 'lowering_plan': 'small component bbox masks only'}, {'task_id': 285, 'compiler_lane': 'L2_local_predicate_sparse_fill', 'strict_cost': 151996, 'gain_to_600': 5.534679828476525, 'gain_to_250': 6.410148565830427, 'candidate': 'zero_holes_majority_border_color', 'status': 'partial', 'train_pass': 0, 'train_examples': 3, 'test_pass': 0, 'test_examples': 1, 'arc_pass': 0, 'arc_examples': 261, 'total_pass': 0, 'total_examples': 265, 'pass_rate': 0.0, 'avg_pred_changed_cells': 0.018867924528301886, 'fail_reasons': '{"ok": 265}', 'lowering_plan': 'component hole mask plus neighboring color vote'}]

## 判断

full passがあればloweringへ進む。full passが少ない場合でも、partial上位をdecision tree合成の素材として使う。
