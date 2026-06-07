# exp121_task366_cast_suffix_extraction

## Hypothesis

task366のfinal-shape mask/type intermediateは、graph output dtypeへ `Cast -> output` するだけで正しい出力として使える可能性がある。

## Result

- campaign index: `23`
- candidates: `6`
- status counts: `rejected=6`
- validation counts: `0_pass_1_fail=6`
- accepted tasks: `[]`
- local delta: `0.000000`

## Interpretation

output dtypeに合わせたCastで静的型問題は解けたが、全候補がexample 0でmismatchした。task366のearly final-shape tensorsは最終outputに意味的に近くない。

これでtask366の既存artifact harvestingはほぼ打ち切り。続けるならfresh representationだが、過去のfull-grid/sparse primitive costが高いため、#24では別のscore-producing laneを検討する。

## Risk

- leakage risk: low。graph-only extraction。
- overfitting risk: low-to-medium。
