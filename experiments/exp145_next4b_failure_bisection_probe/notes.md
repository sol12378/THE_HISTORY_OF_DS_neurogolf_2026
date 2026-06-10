# exp145_next4b_failure_bisection_probe

## Hypothesis

task018修復後、残るpublic-zero候補をさらに絞るため、次点consensus group `032/050/016/031` をfail-stub probeで測る。

## Result

- status: `probe_complete_all_alive`
- targets: `[32, 50, 16, 31]`
- expected_drop_if_all_alive: `70.57622595348958`
- expected_lb_if_all_alive_from_exp127: `5859.973774046511`
- zip sha256: `2cb9f429bb818d1f41678f94796595e9ff127b8d79e98d8ee1d740b83be48910`

## Target Validation

- task032: `0_pass_1_fail`, reason `mismatch example 0`
- task050: `0_pass_1_fail`, reason `mismatch example 0`
- task016: `0_pass_1_fail`, reason `mismatch example 0`
- task031: `0_pass_1_fail`, reason `mismatch example 0`

## Decision

Kaggle ref `53521106` として提出し、public LB は `5859.97`。exp127からのdrop `70.58` は all-alive期待drop `70.57623` と一致した。

task032/050/016/031 はpublic-scoring alive。public-zero repair suspectから除外し、次groupへ進む。

## Leakage / Overfitting Risk

low: deliberate fail-stub probe; no private labels used.
medium: consumes a submission and only measures public scoring behavior for a group.
