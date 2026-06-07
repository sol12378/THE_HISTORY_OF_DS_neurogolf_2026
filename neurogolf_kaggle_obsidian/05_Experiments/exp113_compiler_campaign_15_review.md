# exp113_compiler_campaign_15_review

## Purpose

compiler campaign #11〜#15が本当にLB向上のために有意義だったかをレビューする。

## Result

- #11〜#15 local delta: `+0.1239388580395987`
- #1〜#15 local delta: `+0.1239388580395987`
- current submit-safe working local: `6282.93615685804`
- accepted unique tasks: `[48, 184, 187, 207, 263, 316, 394]`

## Verdict

`lb_useful_but_not_sufficient`

focused artifact surgeryはLB向上に有意義だった。#1〜#10のdelta 0から、#11〜#13でsubmit-safe localを実際に上げたため、少なくともpost-pass / calibration laneとして価値がある。

ただし、#13で逓減し、#14でtask185 suffix extractionが失敗した。7500へ向かうには、既存artifactの局所削りではなく、rule-hit taskのfresh fused compilerが必要。

## Next Policy

#16〜#20はfresh fused loweringを主条件にする。

- task185 lattice homogeneous-2x2
- task366 object-marker copy
- task365 rectangle selector

focused surgeryはbundle後の標準post-passに留める。

## Risk

- leakage risk: low for accepted graph surgery。
- overfitting risk: low-to-medium。#11〜#13 bundleはLB未較正。
