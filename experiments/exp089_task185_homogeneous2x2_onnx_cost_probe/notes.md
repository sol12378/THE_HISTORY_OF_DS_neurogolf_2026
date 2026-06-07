# exp089_task185_homogeneous2x2_onnx_cost_probe

## 目的

`task185` のcore ruleである4x4 lattice matrix -> 3x3 homogeneous 2x2 block判定が、ONNX cost `<=600` に入るかを測る。

## 結果

`result.json` / `cost_probe.csv` を参照。

## 判断

core compressionが軽ければ、残る課題はdynamic lattice extraction。重ければ、Conv以外の同色判定または既存artifact surgeryへ切り替える。

## Risk

- leakage risk: low。
- overfitting risk: medium。static Sliceはproxyであり、提出候補ではない。
