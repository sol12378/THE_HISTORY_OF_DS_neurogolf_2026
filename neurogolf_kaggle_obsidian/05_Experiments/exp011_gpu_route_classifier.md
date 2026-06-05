# exp011_gpu_route_classifier

## 目的

軽量CNN + tabular特徴をGPUで学習し、各taskの `route` / `template_family` rankingを作る。

## 結果

- status: `trained`
- device: `cuda`
- final valid accuracy: `0.9487`
- macro F1: `0.8551`
- samples: `9646`
- split: `task_id % 5 == 0` をvalid

## 成果物

- `route_predictions.csv`
- `metrics_by_class.csv`
- `split_manifest.csv`
- `model.pt` はGit ignore対象

## 判断

学習器は採用判定ではなく、候補生成順の決定に使う。最終採用は必ずONNX static validation、sample validation、official-like costで決める。

## Risk

- teacher labelはrule-basedなので、実際の最適templateとはズレる。
- task数が400と少ないため、route classifier単体の精度を過信しない。
