# exp045_rule_searcher_best_practices_7700

## 目的

この実験はスコアを直接探すものではなく、7700到達がrule searcherの整備で説明可能かを検証するための設計証明である。

## 仮説

`exp041_task145_deeper_mul_chain` のlocal upper boundを固定baseとし、全400 taskをfamily別rule searcherで説明可能programへ置換し、平均的に `cost<=250` 相当まで落とせれば7700を超える。

## 結果

- base local estimate: `6480.302478`
- all400 `cost<=250` projection: `7834.734477`
- margin over 7700: `+134.734477`
- phase1 top100 `cost<=9000`: `6613.864741`
- phase2 top200 `cost<=3000`: `6876.375184`
- phase3 top320 `cost<=1000`: `7309.607319`

## Family Best Practices

| family | task count | gain to cost<=250 | best practice |
|---|---:|---:|---|
| `signature_lookup_current` | 196 | `+808.635` | signatureを説明変数へ分解し、浅いdecision tree / affine hash / color-role ruleへ置換する。arc-gen label丸暗記は禁止。 |
| `crop_resize` | 85 | `+216.206` | fixed crop, anchor crop, bbox cropを分け、ONNX前にshape ruleとindex sizeをcost gateへ通す。 |
| `sparse_edit_or_object_completion` | 57 | `+139.461` | changed-cell maskを先に説明し、少数座標だけをemitする。広いmaskはConv/Reduceで閉形式化する。 |
| `point_to_line_pattern` | 28 | `+93.619` | seed color, direction, period, stop conditionをroleで推定し、絶対色より構造不変量を優先する。 |
| `line_grid_fill` | 14 | `+44.760` | complete/incomplete row-column、交点、stop blockerを列挙し、線分単位で正しさを証明する。 |
| `region_partition_fill` | 15 | `+44.443` | flood fillを直接unrollせず、境界線・部屋・内外判定をrow/column prefixや小kernelで閉形式化する。 |

## 解釈

7700は「今すぐ提出できるartifact」としては未証明だが、「family別rule searcher + cost-aware loweringで必要利得を満たせる」という開発命題としては成立する。特に `signature_lookup_current` の196 taskを説明可能ruleへ圧縮することが最大のボトルネックかつ最大の利得源である。

## Leakage / Overfitting Risk

- leakage risk: 中〜高。現baseにはlookup-derived artifactが残る。exp045はこれを提出候補にするのではなく、置換対象として明示した。
- overfitting risk: 中。all arc-genをprojection/proofに使うため、提出候補化にはfamily holdoutとofficial utility validationが必要。

## 次アクション

1. `signature_lookup_current` のlookup compression minerを作る。
2. `crop_resize` のshape-specialized compilerを作り、full-grid GatherND bboxを避ける。
3. `region_partition_fill` / `point_to_line_pattern` は正解rule探索より先にclosed-form lowering skeletonを用意する。
