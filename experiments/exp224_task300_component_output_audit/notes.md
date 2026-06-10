# exp224_task300_component_output_audit

## 目的

exp223で最上位候補になったtask300について、出力がどの入力componentのcrop/maskに対応するか、単純rank selectorで説明できるかを監査する。

## 結果

- baseline_cost: `77546`
- ambiguous_or_no_exactish: `0`
- match_type_hits: `{'crop_exact': 267, 'mask_colorized': 267}`
- selector_rank0_hits: `{'rank_size_desc': 267, 'rank_area_desc': 212, 'rank_r0_asc': 91, 'rank_c0_asc': 95, 'rank_color_asc': 86}`

## 判断

rank0 selectorが全例を説明し、出力がcomponent crop/maskに一致するなら次はPython rule化とcost proxy。そうでなければcomponent選択が複雑で、別候補へ移る。
