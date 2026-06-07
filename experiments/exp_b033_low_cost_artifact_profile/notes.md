# exp_b033_low_cost_artifact_profile

## 目的

current submit-safe best `exp_b025` から、cost<=250/600に近いartifactのONNX構造を抽出し、compilerが真似るべき低cost patternを作る。

## 結果

- cost<=250: `19` tasks
- cost<=600: `27` tasks
- cost<=2000: `83` tasks

## 判断

低cost exemplarはstatic Slice/Gather/Identity、小Conv、低param shape/index patternに寄っている。次はobject-anchor cropとsmall Conv/local-mask compilerを優先し、region-fill unrollは続けない。
