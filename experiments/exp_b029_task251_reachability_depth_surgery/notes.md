# exp_b029_task251_reachability_depth_surgery

## 目的

task251既存artifactのreachability depthを削れるか確認する。

## 結果

- candidates: `7`
- valid depth reductions: `0`
- best partial: `safe_name_58` -> `259_pass_1_fail`

## 判断

depth削減は全候補reject。既存artifactのreachability depthはarc-gen上ほぼ必要。次は深さを削るのではなく、closed-form/rectangle-specific maskへ表現を変える。
