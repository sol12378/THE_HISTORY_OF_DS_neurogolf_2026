# exp239_task271_component_output_audit

## 目的

exp223で3x3固定かつone_component_shape `266/267` だったtask271について、component crop/maskまたはmax-color/largest proxyで内容まで説明できるか監査する。

## 結果

- baseline_cost: `28991`
- output_shapes: `{'3x3': 267}`
- output_color_count_hist: `{2: 267}`
- binary_signature_count: `1`
- counters: `{'max_color_crop_top_left': 28, 'max_color_mask_top_left': 0, 'largest_crop_top_left': 106, 'largest_mask_top_left': 0, 'any_same_shape_input_crop_exact': 267, 'component_shape_match': 266, 'component_exactish_unique': 218, 'component_exactish_any': 218, 'component_crop_exact': 218, 'component_mask_binary': 0, 'component_mask_colorized': 0}`
- selector_rank0_hits: `{'rank_area_desc': 80, 'rank_color_asc': 78, 'rank_size_desc': 72, 'rank_r0_asc': 58, 'rank_c0_asc': 50}`

## 判断

内容一致がfull/nearでselectorが単純ならrule/cost probeへ進む。shapeだけなら別候補へpivotする。
