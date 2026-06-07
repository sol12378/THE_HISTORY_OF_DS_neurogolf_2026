# exp_b022_public_notebook_intelligence

## 目的

ユーザー指定のKaggle notebookを取得し、blend/sourceとしてだけでなくrule/lowering知識として使えるように分類する。

## Notebook Intelligence

- `karnak_all_task_description`: task_description_taxonomy / all-task description analysisをrule family/lane分類に使う / next: family taxonomy / task description prompts / rule-search target queue
- `konbu17_blended_till_4_27`: blend_source_inventory / blend構成とsource採用順をstrict seed / LB calibration観点で読む / next: source reliability / task-level delta / duplicate-source audit
- `magmacot_new_blending`: blend_source_inventory / new blendingのsource構成をstrict seedと比較して信頼できるdelta候補を探す / next: blend source audit / micro-delta candidate selection
- `massimilianoghiotto_convolution_series_part_3`: conv_lowering_patterns / Convolution seriesからsmall Conv/Pad/Slice loweringのbest practiceを抽出する / next: small Conv kernels / padding discipline / local predicate lowering
- `needless090_4250`: baseline_source_inventory / 低スコアだが単純・低riskなsourceとして、strict seed差分や初期taskを確認する / next: low-risk source audit / baseline duplicate check
- `seddiktrk_surgical_precision`: onnx_surgery_precision / surgical precision/parameter reductionから安全なgraph surgery候補を抽出する / next: initializer pruning / dtype or parameter reduction / static safety guardrails
- `vyanktesh_multi_source`: multi_source_solver_patterns / multi-source ONNX solverのtask選択・Slice/Gather/Pad実装を読む / next: source reliability plus cheap Slice/Gather/Pad lowering patterns

## Decision

公開notebookはそのままcopyするのではなく、strict/full-arc validationと小delta LB calibrationを通した知識源として使う。
