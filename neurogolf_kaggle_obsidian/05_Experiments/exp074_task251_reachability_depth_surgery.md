# exp074_task251_reachability_depth_surgery

## 目的

`task251` の既存artifactが持つreachability propagation depthを浅くし、full-arc validationを保ったままcostを削れるか確認する。

## 結果

- base: `exp_b025_submit_safe_delta_union`
- task251 base cost: `100580`
- candidates: 7
- improved candidates: 0
- all shallower outputs rejected
- shallowest near-pass: `depth6_safe_name_58`, validation `259_pass_1_fail`

## 解釈

既存artifactの最終reachability出力 `safe_name_61` は、arc-gen上ほぼ必要。depthを1段削ってもfailするため、単純なgraph surgeryではcostを落とせない。

## 判断

提出なし。task251 ruleは保持するが、score-producing次手は別のclosed-form connectivity lowering、または `task085` erase/mask cleanへ移す。
