# exp252_small_output_fullarc_bypass_sweep

## 目的

small-output高cost候補に対し、既存artifactの1-pass node bypass surgeryをfull-arc gate付きで試し、即提出可能なcost改善があるか確認する。

## 結果

- targets: `185,253,22,271,79,242,130,134,355,153,391,346,274,263`
- generated_candidate_count: `517`
- sample20_improved_count: `2`
- fullarc_selected_count: `1`
- selected task: `153`
- task153: `Reshape_node7_to_input0`, cost `11212 -> 10947`, points `15.675260 -> 15.699179`

## 判断

task153はfull-arc passかつ微小改善。exp234 current bestへ載せて提出候補化する。

## リスク

graph surgeryはsemantics-preserving候補だが、privateはLB較正で確認する。
