# exp045_rule_searcher_best_practices_7700

## 目的

この実験はスコアを直接探すものではなく、7700到達がルール探索器の整備で説明可能かを検証するための設計証明である。
既存のbest localを固定し、400 taskをfamily別に分解して、必要なtarget cost、best practice、禁止lowering、受理基準を明文化する。

## 仮説

全400 taskのうち、artifact/lookup依存をfamily別DSL programへ置換し、平均的にcost<=250相当まで落とせれば7700を超える。
過去実験の失敗から、正しいルールを見つけるだけでは不十分であり、ONNX loweringのcost gateを探索器に組み込む必要がある。

## Projection

| phase | scope | target cost | projected local | gap to 7700 |
|---|---:|---:|---:|---:|
| `phase1_6500_bridge` | 100 | 9000 | 6613.865 | 1086.135 |
| `phase2_7000_submit_safe` | 200 | 3000 | 6876.375 | 823.625 |
| `phase3_7400_private_like` | 320 | 1000 | 7309.607 | 390.393 |
| `phase4_7700_proof` | 400 | 250 | 7834.734 | -134.734 |

## Family Best Practices

| family | tasks | gain_to_250 | searcher | best practice |
|---|---:|---:|---|---|
| `signature_lookup_current` | 196 | 808.635 | lookup compression miner | signatureを説明変数へ分解し、浅いdecision tree/affine hash/color-role ruleへ置換する。arc-gen labelの丸暗記は禁止。 |
| `crop_resize` | 85 | 216.206 | object-anchor crop/resize synthesizer | fixed crop, anchor crop, bbox cropを分けて探索し、ONNX前にshape ruleとindex sizeをcost gateへ通す。 |
| `sparse_edit_or_object_completion` | 57 | 139.461 | sparse edit/object completion enumerator | changed-cell maskを先に説明し、少数座標だけをemitする。広いmaskはConv/Reduceで閉形式化する。 |
| `point_to_line_pattern` | 28 | 93.619 | seed-to-pattern grammar | seed color, direction, period, stop conditionをroleで推定し、絶対色より構造不変量を優先する。 |
| `line_grid_fill` | 14 | 44.760 | line/ray/grid grammar | complete/incomplete row-column、交点、stop blockerを列挙し、線分単位で正しさを証明する。 |
| `region_partition_fill` | 15 | 44.443 | region/room-fill synthesizer | flood fillを直接unrollせず、境界線・部屋・内外判定をrow/column prefixや小kernelで閉形式化する。 |
| `same_shape_global_transform` | 1 | 5.696 | global transform and compact surgery | flip/rotate/transpose/recolorを最小ONNXと既存artifact surgeryの両方で比較する。 |
| `color_map` | 4 | 1.613 | color role normalizer | 単独color mapだけでなく他familyの前処理として扱い、consistent role mapのみ採用する。 |

## 証明としての意味

- base local: `6480.302478`
- cost<=250 projection: `7834.734477`
- 7700 margin: `134.734477`
- exp042 rule-search proxy delta: `16.229076`
- exp042 vs exp040 surgery multiple: `319.12x`

これは「今すぐ7700のsubmissionがある」証明ではない。証明しているのは、7700に必要な利得が、局所surgeryではなくfamily別ルール探索器とcost-aware loweringの改善量として表現できる、という開発命題である。

## Leakage / Overfitting Risk

- leakage risk: 中〜高。baseにはsignature lookup由来のlocal upper boundが残るため、Phase2以降はlookup compressionで説明可能ruleへ置換する。
- overfitting risk: 中。all arc-gen exactをproofに使うが、提出候補はfamily holdout、公式utility、private-like validationを通す。

## 次アクション

1. `signature_lookup_current` をまず圧縮対象にし、lookupを説明可能なtree/ruleへ置換する。
2. `crop_resize` はbboxをfull-grid GatherNDにしないshape-specialized compilerを作る。
3. `region_partition_fill` と `point_to_line_pattern` は正解ルールより先にcost gate付きlowering skeletonを用意する。
