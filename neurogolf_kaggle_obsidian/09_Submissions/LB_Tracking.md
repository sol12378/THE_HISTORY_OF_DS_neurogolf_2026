# LB Tracking

| Date | Submission | Source Exp | CV | LB | Gap | Notes |
|---|---|---|---:|---:|---:|---|
| 2026-06-05 | ref 53383536 / submission.zip | exp001_baseline | task087 local all pass | 14.50 | -0.007 | single task087 rot180 ONNX; local estimate 14.507; status COMPLETE |
| 2026-06-06 | ref 53414511 / submission.zip | exp041_task145_deeper_mul_chain | 6480.302478 local upper | 3417.71 | -3062.59 | user-requested submit; status COMPLETE. Large local/LB collapse confirms signature lookup/public-artifact overfit risk and invalidates exp016-041 local estimate as submit-safe CV. |
| 2026-06-06 | ref 53414978 / submission.zip | exp005_top_cost_rewrite_strict / exp_b001 calib_001 | 6282.230228 strict full-arc pass | 5929.89 | -352.34 | strict seed resubmit; status COMPLETE. Establishes healthier but still nonzero local/LB calibration gap after exp041 collapse |
| 2026-06-06 | ref 53415099 / submission.zip | unknown / likely strict-seed duplicate | unknown | 5929.89 | unknown | verified via `python -m kaggle competitions submissions -c neurogolf-2026`; no description. Same LB as exp005 strict seed, supporting reproducible strict baseline calibration. |
| 2026-06-06 | ref 53417203 / submission.zip | exp_b021_strict_seed_exp038_micro_delta_submit | 6282.432095 | 5930.02 | -352.41 | strict seed + exp038 tasks 62/145/255/268 micro delta; all full-arc pass; LB +0.13 vs strict seed |
| 2026-06-06 | ref 53417088 / submission.zip | exp066_task020_correctness_onnx_lowering | 6282.439960 strict seed + task020 explicit rule delta | 5930.10 | -352.34 | official-valid nonlookup task020 improvement; task020 cost `90133 -> 73080`, local delta `+0.209732`, LB delta vs exp005 `+0.21`, matching local. |
| 2026-06-06 | ref 53417598 / submission.zip | exp068_seddik_style_strict_scalarization | 6282.610350 exp066 + Seddik-style post-pass | 5930.27 | -352.34 | 34 full-arc-safe graph surgery improvements; local delta vs exp066 `+0.170391`, LB delta `+0.17`, matching local. |
| 2026-06-06 | ref 53417752 / submission.zip | exp_b024_safe_uniform_initializer_scalarization | 6282.257713 strict seed + safe uniform scalarization | 5929.92 | -352.34 | 8 full-arc-safe strict-seed scalarization improvements; local delta `+0.027485`; LB delta vs strict seed `+0.03`, matching local. |
| 2026-06-06 | ref 53418067 / submission.zip | exp_b025_submit_safe_delta_union | 6282.812218 exp068 + exp_b021 union | 5930.40 | -352.41 | exp068 plus full-arc-safe graph surgery tasks 62/145/255/268; local delta over exp068 `+0.201867`; LB delta over exp068 `+0.13`; current submit-safe best. |
