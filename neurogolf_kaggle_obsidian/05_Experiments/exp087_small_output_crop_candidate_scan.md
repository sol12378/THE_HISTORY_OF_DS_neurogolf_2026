# exp087_small_output_crop_candidate_scan

## 目的

`exp086` のarea floorから、`max_output_area <= 14` のcrop/shape taskを `cost<=600` 候補として全400 taskから抽出する。

## 結果

- task count: `400`
- P0 cropish count: `43`
- P1 cropish count: `3`
- top candidates: `task185`, `task022`, `task300`, `task271`, `task355`, `task079`, `task184`, `task346`, `task242`, `task391`

## Decision

`task365` は大幅削減候補として保持するが、250〜600級の主crop laneは最大出力面積が小さいP0/P1 taskへ移す。

## Risk

- leakage risk: low。shape統計とqueue metadataのみ。
- overfitting risk: low。提出候補は生成していない。
