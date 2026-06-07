# exp054_signature_lookup_family_taxonomy

## 目的

`signature_lookup_current` 196 taskを、250〜600 cost帯へ落とすための量産compiler laneへ分類する。

## 結果

- profiled tasks: 196
- total gain to cost<=600: 834.211350
- total gain to cost<=250: 1005.803223

## Compiler Lane Summary

| lane | tasks | gain<=600 | gain<=250 | top tasks |
|---|---:|---:|---:|---|
| L4_shape_crop_resize | 62 | 279.254 | 333.534 | 366 158 77 396 64 205 370 216 29 109 |
| L2_local_predicate_sparse_fill | 65 | 272.801 | 329.706 | 133 173 286 285 5 13 358 110 382 204 |
| L3_object_move_or_erase | 31 | 135.176 | 162.316 | 71 383 202 18 85 34 128 25 69 234 |
| L5_general_sparse_completion | 20 | 75.303 | 92.813 | 54 208 11 70 330 182 368 44 156 163 |
| L6_task_specific_or_defer | 10 | 38.438 | 47.192 | 253 351 400 22 57 271 189 79 242 130 |
| L1_static_sparse_background_fill | 8 | 33.239 | 40.242 | 251 20 37 363 335 333 246 50 |

## 解釈

最大familyの中でも、最初に狙うべきは同shapeで変更が疎な背景fill系である。これらはfull-grid処理を避け、local predicate / object-role / small maskへ落とせる可能性が高い。

shape/crop系はgainが大きいが、過去のfull-grid bbox loweringが失敗しているため、constant shape ruleまたは小さいshape-conditioned Slice/Gatherに限定する。
