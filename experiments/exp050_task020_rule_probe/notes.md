# exp050_task020_rule_probe

## 目的

task020のsignature lookup teacherを明示ruleへ圧縮するため、入出力差分とcomponent geometryを調べる。

## 結果

- examples: 266
- hint counts: {'same_shape': 266, 'sparse_edit': 266, 'single_target_color_3': 51, 'only_fill_background': 266, 'nonzero_bbox_preserved': 266, 'single_target_color_2': 49, 'single_target_color_4': 59, 'single_target_color_8': 56, 'single_target_color_1': 51}
- shape pairs: {('(10, 10)', '(10, 10)'): 266}
- candidate direction: same-shape sparse object/background edit

## 解釈

この段階ではONNXは出さない。`example_summary.csv` と `component_summary.csv` から、changed-cell maskを説明する幾何ruleを作る。

## 次

component bbox/size/colorに基づく候補ruleを列挙し、train + holdout arc-genで評価する。
