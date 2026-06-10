# exp223_small_output_mask_shape_inventory

## 目的

小出力cropish候補について、出力shapeが入力bbox/component/color-countなどの簡単なshape特徴で説明できるか、またbinary mask/templateとして安定しているかを棚卸しする。

## 結果

- task_count: `58`
- high-cost shape full-hit:
  - task300 baseline `77546`, `one_component_shape 267/267`
  - task174 baseline `52019`, `one_component_shape 266/266`
  - task130 baseline `15948`, `one_component_shape 265/265`
- stable binary high-cost:
  - task253 baseline `48269`, binary signature count `1`
  - task355/task346など既知1x1系

## 判断

task300を最優先で個別監査する。shape full-hitはshape/component supplier候補。

## リスク

- leakage risk: low
- overfitting risk: low
