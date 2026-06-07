# exp046_task031_small_bbox_compiler

## 仮説

exp045で見えた既存artifact型のsmall dynamic gather bboxを自前compilerとして再合成すれば、task031の既存artifactに近い、またはそれ以下のcostを達成できる。

## 結果

- status: no_cost_gain
- baseline cost: 18616
- candidate cost: 50589
- validation: 266_pass_0_fail
- score delta: 0.000000
- reason: candidate cost is not lower than baseline

## 解釈

full-grid `GatherND` ではなく最大bboxサイズだけを `Gather` する形にした。これで既存artifactの構造をcompilerとして再現できるかを確認する。

## 次

改善しない場合でも、このtemplateを他のbbox系taskへ適用し、既存artifactより高いtaskだけを狙う。
