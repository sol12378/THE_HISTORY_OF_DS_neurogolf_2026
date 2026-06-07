# exp113_compiler_campaign_15_review

## Purpose

compiler campaign #11〜#15が本当にLB向上のために有意義だったかを問い直す。

## Result

- #11〜#15 local delta: `+0.1239388580395987`
- new submit-safe working local: `6282.93615685804`
- accepted unique tasks: `[48, 184, 187, 207, 263, 316, 394]`
- accepted unique task count: `7`

## Interpretation

#11〜#13のfocused artifact surgeryは、#1〜#10のlocal delta 0から明確に改善したため有意義だった。特にfull-arc gated graph surgeryは過去のmicro-delta提出でLB転移実績がある。

ただし、#13でdeltaが `+0.013967` まで逓減し、#14ではtask185の中間output/subgraph extraction候補が0だった。したがって、既存artifactの局所削りだけではLB 7500には届かない。

## Next Policy

#16〜#20はfresh fused loweringを必須にする。

- task185: lattice homogeneous-2x2を1-Conv以下またはPad最小のfused表現へ落とす。
- task366: object-marker copyで30x30 full-grid `Where` / `ScatterND` を避ける。
- task365: dense rectangle selectorをshape-branch低cost化する。

focused surgeryはsubmit-safe bundle post-pass / LB calibration laneとして残す。

## Risk

- leakage risk: accepted surgeryはlow。future fused loweringはhidden-style validationを必須にする。
- overfitting risk: low-to-medium。#11〜#13 bundleはまだLB未較正なので、提出する場合は小delta較正として扱う。
