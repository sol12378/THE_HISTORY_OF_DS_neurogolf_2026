# exp_b001_rule_replacement_backlog

## 目的

exp_b系列の開始点として、全400 taskを説明可能rule / decision tree / color-role ruleへ置換するためのbacklogを作る。

有用でない、または既存artifactがすでに小さいtaskは、無理な置換ではなくfamily別best practice探索へ回す。

## 仮説

7700への近道はlocal score huntingではなく、submit-safe seedから始めて、説明可能ruleとcost-aware ONNX loweringをfamily単位で較正すること。

## 結果

- strict seed: `exp005_top_cost_rewrite_strict`, local `6282.230228`
- teacher/local upper: `exp041_task145_deeper_mul_chain`, local `6480.302478`, LB `3417.71`
- backlog task count: `400`
- family count: `8`
- submission calibration: `calib_001_strict_seed_resubmit` をKaggle提出済み、ref `53414978`、LB `5929.89`。
- calibration gap: strict local `6282.230228` vs LB `5929.89`、差分 `-352.34`。

## Family Summary

| family | tasks | priority | projected gain to phase targets | best practice |
|---|---:|---|---:|---|
| `signature_lookup_current` | 196 | P0 | `+299.621` | teacher artifactはoracleとしてprofileし、入出力差分から人間可読なinvariantだけを採用する。 |
| `crop_resize` | 85 | P1 | `+118.327` | bboxを見つけてもすぐONNX化せず、index table sizeとactivation memoryを先に見積もる。 |
| `sparse_edit_or_object_completion` | 57 | P1 | `+73.363` | 変更セル数が少ないことを証明してから、全grid処理を避けるloweringだけをemitする。 |
| `point_to_line_pattern` | 28 | P1 | `+42.822` | seed、方向、period、stop conditionを分けて探索し、絶対色ではなくroleで記述する。 |
| `line_grid_fill` | 14 | P1 | `+23.495` | 線分単位の閉形式maskを作り、unrollしない。 |
| `region_partition_fill` | 15 | P1 | `+19.881` | flood fillの正しさを利用しても、実装は閉形式の部屋maskへ落とす。 |
| `same_shape_global_transform` | 1 | P2 | `+2.112` | 既存artifactが小さい場合はrule ONNXよりsurgeryの方を優先する。 |
| `color_map` | 4 | P2 | `+1.613` | 単独改善より他familyの前処理として使う。 |

## Submission Calibration

1. `calib_001_strict_seed_resubmit`: exp005 strict seedを再提出し、local/LBの基準線を作る。
2. `calib_002_single_family_delta`: 最初の受理済みrule-lowering candidateを1 taskだけ差し替え、local deltaとLB deltaの対応を見る。
3. `calib_003_p0_lookup_replacement_bundle`: P0 lookup-compressed rulesだけをbundle化し、LB collapseしないか確認する。
4. `calib_004_mixed_family_growth`: trusted calibrated familiesだけを段階追加する。

## Leakage / Overfitting Risk

- leakage risk: 中。teacher artifactはdiagnostic oracleとしてのみ使い、rule化されるまでsubmit-safeとは見なさない。
- overfitting risk: 中。all arc-genだけで採用せず、Kaggle calibration submissionでlocal/LB対応を確認する。

## 次アクション

1. P0上位taskからrule minerを開始する。直近の候補は `task020` と P0上位の `task286`。
2. 受理済みruleだけを安いONNX loweringへ落とし、single-task差し替えsubmissionでlocal/LB対応を見る。
3. local/LBの通常gapはまず `-352.34` を基準線にし、exp041型の大崩壊を起こすfamilyを早期検出する。
