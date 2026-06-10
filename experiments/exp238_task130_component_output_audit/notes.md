# exp238_task130_component_output_audit

## 目的

exp223でshape-full候補になったtask130について、task300と同様にcomponent crop/maskまたはmax-color proxyで出力内容まで説明できるか監査する。

## 結果

- baseline_cost: `15948`
- output_shapes: `{'3x3': 265}`
- output_color_count_hist: `{1: 37, 2: 44, 3: 35, 4: 22, 5: 43, 6: 31, 7: 29, 8: 24}`
- binary_signature_count: `168`
- counters: `{'max_color_crop_top_left': 0, 'max_color_mask_top_left': 0, 'largest_crop_top_left': 0, 'largest_mask_top_left': 0, 'any_same_shape_input_crop_exact': 5, 'component_shape_match': 265, 'component_exactish_unique': 7, 'component_exactish_any': 8, 'component_crop_exact': 0, 'component_mask_binary': 8, 'component_mask_colorized': 0}`
- top_exact_crop_offsets: `[('2,4', 2), ('2,2', 2), ('4,2', 1)]`

## 判断

component exactishまたはmax-color/largest proxyがfullならcost probeへ進む。shapeのみ一致で内容が合わない場合は、task130は別途content rule miningが必要。
