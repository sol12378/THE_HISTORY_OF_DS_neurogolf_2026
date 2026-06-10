# exp240_task271_crop_offset_selector_audit

## 目的

task271の3x3 exact crop offsetを単純な入力特徴で選べるか確認する。

## 結果

- exact_offset_count_hist: `{1: 267}`
- best selector: `comp_area_desc_tl/clamped 176/267`
- `comp_size_desc_tl/clamped`: `106/267`
- `nz_bbox_tl`: `31/267`

## 判断

exact cropは各例一意だが、単純なbbox/component top-left selectorではfullにならない。固定binary template windowとして候補を絞る。
