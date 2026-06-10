# exp160_task158_dtype_rewrite_inspection

## 目的

exp159でdtype post-pass候補になったtask158について、full-grid Cast-to-FLOAT の消費先を確認し、semantic-preservingなdtype rewriteが可能そうか判断する。

## 結果

- cast_to_float_fullgrid_count: `18`
- rewrite_hint_counts: `numeric_consumers_need_float: 18`
- すべて `BOOL -> FLOAT` Cast の出力が `Sum` に入っていた。

## 解釈

task158のCastはboolean maskを数値カウントするためにFLOATへ変換している。これは `Sum` の入力dtype要件に関わるため、単純にBOOL/UINT8のまま保持するrewriteは安全ではない。exp159のheuristic savingは大きかったが、直接dtype removal候補としては弱い。

## 判断

task158の直接dtype rewriteは保留。次は `Where` 条件やmask selectionに近い候補を優先してinspectする。候補としてはtask366/328/206/338など、`cast_from_boolish_to_float` と `where_full_grid` が同時にあるtaskを再確認する。

## リスク

- leakage risk: low。graph構造検査のみ。
- overfitting risk: low-to-medium。rewrite可否の局所判断であり、実candidateはまだない。
