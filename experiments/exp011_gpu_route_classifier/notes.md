# exp011_gpu_route_classifier notes

## 目的

GPUで軽量CNN + tabular特徴のroute classifierを学習し、top cost taskに対するtemplate候補生成順のrankingに使う。

## 結果

- status: `trained`
- device: `cuda`
- torch: `2.12.0+cu126`
- samples: `9646`
- train/valid split: task_id group split (`task_id % 5 == 0` がvalid)
- final valid accuracy: `0.9487`
- macro F1: `0.8551`

## Risk

- leakage risk: 中。teacher labelはrule-basedで、classifierの出力は採用判定ではなく候補生成順にだけ使う。
- overfitting risk: 中。route分類は粗い補助タスクなので、最終採用はONNX validationとofficial-like costで決める。
- `model.pt` はGit管理外。再現情報は `result.json`、`route_predictions.csv`、`metrics_by_class.csv` に残す。
