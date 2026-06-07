# exp_b022_public_notebook_intelligence

## 目的

ユーザー指定のKaggle notebook 7本を取得し、blend/sourceとしてだけでなく、rule/lowering/graph surgery/task taxonomyの知識源として使えるように分類する。

## 取得notebook

- `massimilianoghiotto/convolution-series-part-3`
- `konbu17/neurogolf-2026-blended-till-4-27`
- `vyankteshdwivedi/neurogolf-multi-source-onnx-solver`
- `seddiktrk/surgical-onnx-precision-parameter-reduction`
- `karnakbaevarthur/all-task-description-analysis`
- `magmacot/neurogolf-new-blending`
- `needless090/neurogolf-4250`

## 分類

- `massimilianoghiotto_convolution_series_part_3`
  - category: `conv_lowering_patterns`
  - use: small Conv / Pad / Slice loweringのbest practice抽出
- `konbu17_blended_till_4_27`
  - category: `blend_source_inventory`
  - use: source reliability / task-level delta / duplicate-source audit
- `vyanktesh_multi_source`
  - category: `multi_source_solver_patterns`
  - use: multi-source task選択とcheap Slice/Gather/Pad lowering patterns
- `seddiktrk_surgical_precision`
  - category: `onnx_surgery_precision`
  - use: initializer pruning / dtype or parameter reduction / static safety guardrails
- `karnak_all_task_description`
  - category: `task_description_taxonomy`
  - use: family taxonomy / task description prompts / rule-search target queue
- `magmacot_new_blending`
  - category: `blend_source_inventory`
  - use: strict seedと比較するmicro-delta候補探索
- `needless090_4250`
  - category: `baseline_source_inventory`
  - use: low-risk source audit / baseline duplicate check

## Decision

公開notebookはそのままcopyせず、構造化知識として使う。blend系はsource reliability、Conv/surgery系はlowering/graph surgery、description analysisはrule taxonomyに反映する。採用候補は必ずstrict/full-arc validationとmicro-delta LB calibrationを通す。

## Risk

- leakage risk: medium。公開notebookにはLB tuned artifactが含まれ得る。
- overfitting risk: medium-high。blind copyは避け、独立検証されたrule/loweringとして使う。
