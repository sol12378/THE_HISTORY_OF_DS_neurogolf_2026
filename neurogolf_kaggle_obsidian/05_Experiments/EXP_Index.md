# Experiment Index

| Exp | Status | CV / Local | LB | Decision |
|---|---|---:|---:|---|
| exp001_baseline | submitted_complete | task087 local all pass | 14.50 | first API submission baseline |
| exp002_public_blend_6500_fast | below_target | 6282.08 | - | strict public blend did not reach 6500 |
| exp003_rule_audit_compress | compress_banned | 6282.08 | - | Compress is officially excluded |
| exp004_public_blend_relaxed_static | below_target | 6282.08, full validation pass | - | relaxed official-safe filter did not improve |
| exp005_top_cost_rewrite_strict | best_strict | 6282.23 | - | unused initializer prune accepted |
| exp007_gpu_task_taxonomy_and_rewrite_router | diagnostic | taxonomy complete | - | useful for routing |
| exp008_extra_public_sources_strict | no_gain | 6282.08 | - | added public sources, no strict gain |
| exp009_ort_graph_optimization_strict | no_gain | 6282.23 | - | ORT optimization no gain |
| exp010_cuda_gpu_setup | cuda_ready | smoke pass | - | GPU available |
| exp011_gpu_route_classifier | trained | valid acc 0.9487 | - | use for candidate ranking |
| exp012_template_factory_core | below_6500_high_risk | 6296.30 sample-local | - | lookup improved 38 tasks but not submit-safe |
