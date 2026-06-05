# Experiment Summary

## Current Best

| Type | Experiment | CV / Local | LB | Notes |
|---|---|---:|---:|---|
| Baseline | exp001_baseline | task087 local all pass | 14.50 | single task087 rot180 ONNX |
| Best Strict Bundle | exp005_top_cost_rewrite_strict | local estimate 6282.23 | - | strict public blend + unused initializer prune, full validation pass |
| Best Local Upper Bound | exp012_template_factory_core | sample-local estimate 6296.30 | - | GPU-routed template factory with signature lookup; high leakage/overfitting risk, not a submit candidate |

## Experiments

| Exp | Date | Hypothesis | CV / Local | LB | Decision | Notes |
|---|---|---|---:|---:|---|---|
| exp001_baseline | 2026-06-05 | Establish a reproducible NeuroGolf baseline and first ONNX submission loop. | task087 local all pass | 14.50 | submitted_complete | Official spec confirmed; task087 rot180 ONNX submitted via Kaggle API, ref 53383536. |
| exp002_public_blend_6500_fast | 2026-06-05 | Public artifact blend can reach local 6500 under strict static/cost validation. | 6282.08, 400/400 selected | - | below_target | Strict banned-op/static filter rejected many high-score artifacts; no submit. |
| exp003_rule_audit_compress | 2026-06-05 | Allowing Compress may explain the gap to higher public artifacts. | 6282.08 | - | compress_banned | Official utility excludes Compress, so relaxed Compress route is not submit-safe. |
| exp004_public_blend_relaxed_static | 2026-06-05 | Official-safe relaxed static filtering plus full validation can recover 6500. | 6282.08, full validation 400/400 | - | below_target | No gain over exp002; full validation passed for selected bundle. |
| exp005_top_cost_rewrite_strict | 2026-06-05 | Strict graph cleanup can reduce top-cost tasks without rule risk. | 6282.23, full validation 400/400 | - | best_strict | Removed unused initializers on 7 tasks; +0.1545. |
| exp006_private_like_validation | 2026-06-05 | Public artifact risk should be separated from strict local estimate. | analysis only | - | folded_into_notes | Private-like validation direction recorded as next PDCA, not a separate runnable result yet. |
| exp007_gpu_task_taxonomy_and_rewrite_router | 2026-06-05 | Top-cost tasks can be routed into template families before rewrite. | taxonomy for top-cost tasks | - | useful_diagnostic | Top tasks are mostly sparse/object-completion and crop/resize; contact sheet generated. |
| exp008_extra_public_sources_strict | 2026-06-05 | More public artifacts can raise strict blend toward 6500. | 6282.08 | - | no_gain | Added many accessible datasets/notebook outputs; strict best unchanged. |
| exp009_ort_graph_optimization_strict | 2026-06-05 | ONNX Runtime graph optimization can shrink strict selected models. | 6282.23 | - | no_gain | ORT optimization produced no accepted cost reduction. |
| exp010_cuda_gpu_setup | 2026-06-05 | CUDA/PyTorch should be installed first to support GPU-assisted routing. | CUDA smoke pass | - | cuda_ready | torch 2.12.0+cu126, torchvision 0.27.0+cu126, RTX 2080 SUPER confirmed. |
| exp011_gpu_route_classifier | 2026-06-05 | GPU classifier can rank template families for top-cost rewrite. | valid accuracy 0.9487, macro F1 0.8551 | - | trained | CUDA training complete; model.pt ignored, route_predictions/metrics saved. |
| exp012_template_factory_core | 2026-06-05 | GPU-routed template factory can replace expensive strict models with cheaper static ONNX. | 6296.30 sample-local, +14.07 | - | below_6500_high_risk | Signature lookup improved 38 tasks but is high leakage risk and still below 6500; no submit. |

## Current Assessment

- CUDA is usable and should remain part of the workflow for route classification, feature mining, and candidate ranking.
- Final ONNX acceptance must stay CPU/official-utility validated.
- 6500 was not reached. The gap is no longer simple public artifact blending; it needs task-specific graph surgery or stronger learned/template solvers.
- Do not submit exp012 without a separate user confirmation and full validation/risk review.
