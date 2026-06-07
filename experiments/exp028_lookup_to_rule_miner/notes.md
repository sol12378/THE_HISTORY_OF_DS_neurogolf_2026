# exp028_lookup_to_rule_miner

## Hypothesis

signature lookup 196件の一部は、train-onlyで抽出できる固定crop、global transform、bbox crop、固定sparse editへ圧縮できる。
この段階ではONNXを出さず、rule候補とproof logを作って次のlowering実験へ渡す。

## Result

- scanned signature lookup tasks: `196`
- validation-pass rule candidates: `0`
- tasks with at least one pass rule: `0`
- positive projected candidates: `0`
- base local estimate: `6479.398825`

## Interpretation

fitにはtrainだけを使い、testとarc-gen先頭20件はvalidation専用にした。これにより、exp016系のarc-gen memorized lookupから説明可能ruleへ戻す入口を作った。

## Next

- passした `fixed_crop_color_map` / `global_transform_color_map` は低cost ONNXへ即接続する。
- `bbox_crop_color_map` は有望だがdynamic bbox loweringが課題なので、exp029でcost-aware loweringを作る。
- projected gainが正の候補だけを、次のONNX campaignへ渡す。
