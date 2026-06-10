# exp137_private_failure_consensus_audit

## Hypothesis

exp136の上位subset候補に頻出するtaskほど、-352 gapを作るprivate failure群である可能性が高い。

## Result

- subset rows used: `20`
- unique candidate tasks: `56`

## Priority Queue

- task013: freq `20`, exact `4`, source `massimilianoghiotto_6254`, route `sparse_edit_or_object_completion`, points `13.3311`, ops `Conv:20 ReduceMax:18 Greater:14 Where:13 Mul:10 Cast:9 ReduceSum:2 Sum:2`
- task002: freq `20`, exact `4`, source `massimilianoghiotto_6254`, route `sparse_edit_or_object_completion`, points `13.5145`, ops `Conv:24 Greater:23 Where:21 Slice:11 Sub:6 Cast:3 Sum:2 ReduceSum:1`
- task029: freq `20`, exact `4`, source `massimilianoghiotto_6254`, route `crop_or_resize`, points `13.5163`, ops `Cast:12 Sub:12 Greater:9 Add:9 ArgMax:8 Mul:7 Gather:6 ReduceMax:5`
- task009: freq `20`, exact `4`, source `massimilianoghiotto_6254`, route `sparse_edit_or_object_completion`, points `13.6948`, ops `MatMul:4 Greater:4 Cast:2 And:2 Transpose:2 GatherND:1 Reshape:1 Or:1`
- task024: freq `19`, exact `4`, source `massimilianoghiotto_6254`, route `sparse_edit_or_object_completion`, points `15.8862`, ops `Cast:7 And:5 Slice:4 Or:4 ReduceMax:3 Not:2 Concat:1 Pad:1`
- task018: freq `18`, exact `4`, source `kojimar_5800_minimal_blend`, route `sparse_edit_or_object_completion`, points `13.3510`, ops `ReduceSum:3 Slice:2 Mul:2 Concat:2 Sub:2 Unsqueeze:1 MatMul:1 Abs:1`
- task004: freq `17`, exact `4`, source `massimilianoghiotto_6254`, route `sparse_edit_or_object_completion`, points `14.0845`, ops `Conv:2 Cast:1 Relu:1`
- task019: freq `19`, exact `3`, source `massimilianoghiotto_6254`, route `crop_or_resize`, points `13.6935`, ops `Cast:4 ReduceSum:4 Add:3 ReduceMax:2 Mod:2 Less:2 Where:2 Gather:2`
- task032: freq `17`, exact `3`, source `massimilianoghiotto_6254`, route `sparse_edit_or_object_completion`, points `16.8670`, ops `Sub:3 ReduceSum:2 Cast:2 Slice:1 Conv:1 Sign:1 Sqrt:1 Greater:1`
- task050: freq `15`, exact `3`, source `afr1ste_5689_artifact`, route `sparse_edit_or_object_completion`, points `15.8436`, ops `Cast:8 CumSum:4 And:3 Slice:2 Or:1 Xor:1 Concat:1 Pad:1`
- task016: freq `15`, exact `3`, source `massimilianoghiotto_6254`, route `sparse_edit_or_object_completion`, points `22.6974`, ops `Gather:1`
- task031: freq `13`, exact `3`, source `massimilianoghiotto_6254`, route `crop_or_resize`, points `15.1682`, ops `ArgMax:4 Where:3 ReduceSum:2 Greater:2 Cast:2 Sub:2 Add:2 Gather:2`
- task021: freq `13`, exact `3`, source `massimilianoghiotto_6254`, route `crop_or_resize`, points `14.1561`, ops `Cast:52 ReduceMax:33 Equal:27 ReduceSum:20 Greater:19 Add:15 And:10 Not:9`
- task014: freq `13`, exact `3`, source `massimilianoghiotto_6254`, route `crop_or_resize`, points `14.4274`, ops `Reshape:7 Add:7 Cast:5 ArgMax:5 Where:3 Gather:2 ReduceMax:2 Sub:2`
- task025: freq `12`, exact `3`, source `franksunp_blended_best`, route `sparse_edit_or_object_completion`, points `13.6004`, ops `Cast:2 ReduceSum:1 MatMul:1 Equal:1 ArgMax:1 Gather:1 Flatten:1 ScatterElements:1`

## Interpretation

上位候補は低番号taskに偏るため、subset-sumだけを信じず、頻度・source・graph構造を合わせて監査queueとして使う。最初のrepair候補は頻出かつ既にrule資産があるtask366、またはsource差し替え可能性が高いcrop/sparse taskを優先する。

## Leakage / Overfitting Risk

診断のみで提出なし。private failureの直接証明ではない。
