# exp294_bundle_full_arc_pass_gate

## Plan

real supplier が `full_arc_pass` 表記の候補を直接 `BundleCandidate` に入れても accuracy gate を通れるようにする。

## Do

`bundle_manager.py` の `BundleCandidate.accepted` で `full_arc_pass` と `*_pass_0_fail` を pass 扱いにした。

## Check

- `full_arc_pass`: accepted
- `266_pass_0_fail`: accepted
- `24_pass_1_fail`: rejected
- `best_total_local_delta`: `0.3285040669720361`

## Act

提出なし。synthetic gate probe のみ。
