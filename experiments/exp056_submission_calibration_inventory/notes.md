# exp056_submission_calibration_inventory

## 目的

低cost化と並行して、local/LBの差を早く縮めるため、既存submission候補を棚卸しする。

## 結果

- candidate experiments: 67
- submission.zipあり: 29
- known LBあり: 3
- submit-ready候補: 0

## 既知LB

| exp | local | LB | gap | ref |
|---|---:|---:|---:|---|
| exp041_task145_deeper_mul_chain | 6480.302477938763 | 3417.71 | -3062.5924779387633 | 53414511 |
| exp001_baseline |  | 14.5 |  | 53383536 |
| exp005_top_cost_rewrite_strict | 6282.230228092811 | 5929.89 | -352.3402280928103 | 53414978 |

## Decision

現時点では、exp005 strict seedが唯一の健全な較正基準。次に提出すべきなのは、object-role/bbox-local compilerで作ったofficial-validなsingle-task deltaであり、no-hit診断実験やteacher lookup bundleではない。

ただし、ユーザー方針に従い、今後full-passかつofficial-validな小deltaが出たら、localだけで温存せず早めにKaggleへ提出してlocal/LB gapを記録する。
