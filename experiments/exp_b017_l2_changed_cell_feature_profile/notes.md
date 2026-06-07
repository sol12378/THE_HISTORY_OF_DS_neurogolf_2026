# exp_b017_l2_changed_cell_feature_profile

## 目的

L2 top taskのchanged cellをpositive/negative cell datasetとしてprofileし、branch/tree合成で使うべきfeatureを特定する。

## 結果

- target tasks: 5
- task profiles: 5
- best profiles: [{'task_id': 173, 'strict_cost': 175243, 'gain_to_250': 6.552467943298192, 'train_examples': 3, 'positive_cells': 23, 'negative_cells': 1850, 'output_color_counts': '{"1": 6, "6": 1, "8": 11, "4": 4, "2": 1}', 'changed_count_by_example': '{"13": 1, "5": 2}', 'best_feature': 'ray_lr', 'best_value': '-1:5', 'best_precision': 1.0, 'best_recall': 0.043478260869565216, 'interpretation': 'feature is useful but needs branch conjunction'}, {'task_id': 133, 'strict_cost': 193923, 'gain_to_250': 6.653755534178206, 'train_examples': 4, 'positive_cells': 89, 'negative_cells': 1436, 'output_color_counts': '{"4": 24, "3": 7, "6": 4, "8": 54}', 'changed_count_by_example': '{"12": 1, "8": 1, "54": 1, "15": 1}', 'best_feature': 'ray_d1', 'best_value': '-4:-3', 'best_precision': 1.0, 'best_recall': 0.02247191011235955, 'interpretation': 'feature is useful but needs branch conjunction'}, {'task_id': 285, 'strict_cost': 151996, 'gain_to_250': 6.410148565830427, 'train_examples': 3, 'positive_cells': 81, 'negative_cells': 6001, 'output_color_counts': '{"1": 9, "2": 14, "3": 15, "4": 28, "8": 11, "6": 4}', 'changed_count_by_example': '{"39": 1, "24": 1, "18": 1}', 'best_feature': 'ray_d1', 'best_value': '-6:1', 'best_precision': 1.0, 'best_recall': 0.012345679012345678, 'interpretation': 'feature is useful but needs branch conjunction'}]

## 判断

高precision featureがあるtaskを優先してbranch/tree minerへ進む。単一featureで足りないtaskは、color-roleやcomponent featureを追加する。
