# exp087_small_output_crop_candidate_scan

## 目的

`exp086` のcost実測から、`max_output_area <= 14` のcrop/shape taskを `cost<=600` 候補として抽出する。

## 結果

- task count: `400`
- P0 cropish count: `43`
- P1 cropish count: `3`

## 判断

`task365` は6x6 area floorで600級が厳しいため、大幅削減候補として保持する。一方で、最大出力面積が14セル以下のcropish taskは、`Slice+Pad` proxyで600に入る可能性があるため次のrule probe対象にする。

## Risk

- leakage risk: low。shape統計とqueue metadataのみ。
- overfitting risk: low。提出候補は生成していない。
