# exp074_task251_reachability_depth_surgery

## 目的

`task251` の既存artifactが持つreachability propagation depthを浅くし、full-arc validationを保ったままcostを削れるか確認する。

## 結果

- base cost: `100580`
- best candidate: `none`
- local delta: `0.000000000`
- submission decision: `no_submit`

## 判断

浅いreachabilityで通るならsingle-task deltaとして提出候補。通らなければ、既存artifactの深さは必要であり、別のclosed-form loweringが必要。
