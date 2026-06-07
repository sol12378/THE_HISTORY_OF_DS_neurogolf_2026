# exp_b001_rule_replacement_backlog

## 目的

exp_b系列の開始点として、全400 taskを説明可能rule / decision tree / color-role ruleへ置換するbacklogを作る。
有用でない、または既存artifactがすでに小さいtaskは、無理な置換ではなくfamily別best practice探索へ回す。

## 仮説

7700への近道はlocal score huntingではなく、submit-safe seedから始めて、説明可能ruleとcost-aware ONNX loweringをfamily単位で較正すること。

## 結果

- strict seed: `exp005_top_cost_rewrite_strict` / local `6282.230228092811`
- teacher/local upper: `exp041_task145_deeper_mul_chain` / local `6480.302477938763` / LB `3417.71`
- backlog tasks: `400`
- family count: `8`

## Family Summary

| family | tasks | projected gain | priority | best practice |
|---|---:|---:|---|---|
| `signature_lookup_current` | 196 | 299.621 | P0 | teacher artifactはoracleとしてprofileし、入出力差分から人間可読なinvariantだけを採用する。 |
| `crop_resize` | 85 | 118.327 | P1 | bboxを見つけてもすぐONNX化せず、index table sizeとactivation memoryを先に見積もる。 |
| `sparse_edit_or_object_completion` | 57 | 73.363 | P1 | 変更セル数が少ないことを証明してから、全grid処理を避けるloweringだけをemitする。 |
| `point_to_line_pattern` | 28 | 42.822 | P1 | seed、方向、period、stop conditionを分けて探索し、絶対色ではなくroleで記述する。 |
| `line_grid_fill` | 14 | 23.495 | P1 | 線分単位の閉形式maskを作り、unrollしない。 |
| `region_partition_fill` | 15 | 19.881 | P1 | flood fillの正しさを利用しても、実装は閉形式の部屋maskへ落とす。 |
| `same_shape_global_transform` | 1 | 2.112 | P2 | 既存artifactが小さい場合はrule ONNXよりsurgeryの方を優先する。 |
| `color_map` | 4 | 1.613 | P2 | 単独改善より他familyの前処理として使う。 |

## Submission Calibration

local/LB差を早く縮めるため、今後は大きなbundleを一気に出さず、strict seed再提出、single-family delta、P0 lookup replacement bundleの順に提出して較正する。

## リスク

- leakage risk: 中。teacher artifactはoracle扱いで、rule化されるまでsubmit-safeとは見なさない。
- overfitting risk: 中。all arc-gen exactだけでなく、family holdoutとKaggle calibration submissionで確認する。

## 次アクション

1. `calib_001_strict_seed_resubmit` を実行し、exp005 strict seedのLB基準を取る。
2. P0上位taskからrule minerを開始する。まず `task020` をteacher profileから説明可能ruleへ圧縮する。
3. 受理済みruleだけを安いONNX loweringへ落とし、single-task差し替えsubmissionでlocal/LB対応を見る。
