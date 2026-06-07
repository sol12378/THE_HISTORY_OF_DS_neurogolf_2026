# exp054_signature_lookup_family_taxonomy

## Hypothesis

最大gain familyである `signature_lookup_current` 196 taskを、250〜600 cost帯へ落とすための量産compiler laneに分解できる。

## Result

- profiled tasks: `196`
- total gain to cost<=600: `834.211350`
- total gain to cost<=250: `1005.803223`

| Lane | Tasks | Gain to 600 | Gain to 250 | Top tasks |
|---|---:|---:|---:|---|
| L4_shape_crop_resize | 62 | 279.254 | 333.534 | 366 158 77 396 64 205 370 216 29 109 |
| L2_local_predicate_sparse_fill | 65 | 272.801 | 329.706 | 133 173 286 285 5 13 358 110 382 204 |
| L3_object_move_or_erase | 31 | 135.176 | 162.316 | 71 383 202 18 85 34 128 25 69 234 |
| L5_general_sparse_completion | 20 | 75.303 | 92.813 | 54 208 11 70 330 182 368 44 156 163 |
| L6_task_specific_or_defer | 10 | 38.438 | 47.192 | 253 351 400 22 57 271 189 79 242 130 |
| L1_static_sparse_background_fill | 8 | 33.239 | 40.242 | 251 20 37 363 335 333 246 50 |

## Decision

最初の実装波は `L1/L2 sparse fill` にする。`L4 shape/crop` はgain最大だが、過去にfull-grid bbox loweringで失敗しているため、strict cost guardと小さいSlice/Gather条件を先に作る。

## Risk

- leakage risk: low。構造分類のみでteacher tableは出していない。
- overfitting risk: medium。all examplesを診断に使っているため、実ruleにはholdout/較正が必要。
