# exp056_submission_calibration_inventory

## Hypothesis

local/LB差を早くなくすには、低cost化実験と並行して、提出候補を小delta単位で棚卸しし、提出すべきものを明確化する必要がある。

## Result

- candidate experiments: `67`
- submission.zipあり: `29`
- known LBあり: `3` in local metadata
- Kaggle CLI verified submissions: `4`
- submit-ready候補: `0`

Kaggle CLI:

| ref | source | LB | note |
|---:|---|---:|---|
| 53383536 | exp001 | 14.50 | baseline |
| 53414511 | exp041 | 3417.71 | high-risk local upper collapse |
| 53414978 | exp005 / exp_b001 calib_001 | 5929.89 | strict seed |
| 53415099 | unknown / likely duplicate strict seed | 5929.89 | no description |

## Decision

現時点では新規に投げるべき安全deltaはない。exp037〜041系はexp041でcollapse済みのlocal-upper lineageなので、追加提出の情報価値は低い。

次に提出するのは、object-role/bbox-local compilerから出るofficial-validなsingle-task delta。localだけで温存せず、早めにKaggleへ投げてlocal/LB gapを記録する。

## Risk

- leakage risk: low for inventory。
- operational risk: submission slotを浪費しないため、no-hit診断実験やteacher lookup bundleは提出しない。
