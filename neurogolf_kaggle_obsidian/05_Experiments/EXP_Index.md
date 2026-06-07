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
| exp013_node_profile_and_rewrite_targets | diagnostic_complete | top100 profiled | - | rewrite targets identified |
| exp014_sparse_object_template_bank | useful_component | 62/71 targets improved | - | gains dominated by signature ScatterND |
| exp015_crop_resize_template_bank | useful_component | 25/29 targets improved | - | gains dominated by signature ScatterND |
| exp016_top100_rewrite_campaign | below_6500_high_risk | 6479.38 sample-local | - | best local upper bound, no submit |
| exp017_program_synthesis_seed_6500 | no_gain | 6479.38 sample-local | - | iterative neighbor fill did not help |
| exp018_neurogolf_dsl_core | pipeline_foundation | inventory complete | - | DSL family registry for 7600 pipeline |
| exp019_boundary_flood_fill_campaign | no_gain | 6479.38 sample-local | - | boundary flood correct but too expensive |
| exp020_diagonal_periodic_lowmem | no_gain | 6479.38 sample-local | - | periodic rule correct but Where-chain too expensive |
| exp021_diagonal_scatternd_lowmem | no_gain | 6479.38 sample-local | - | task398 rule passed but ScatterND lowering too expensive |
| exp022_program_synthesis_pipeline_7600 | pipeline_backlog_ready | backlog ready, base 6479.38 | - | 7600 pipeline backlog and phase gates generated |
| exp023_graph_surgery_exp016 | improved_below_target | 6479.40 sample-local | - | pruned unused initializer from 195 lookup models |
| exp024_ring_depth_dynamic_colormap | no_gain | 6479.38 sample-local | - | task203 ring rule passed but dynamic MatMul cost too high |
| exp025_tile_prefix_blocks_task221 | no_gain | 6479.38 sample-local | - | task221 tile-prefix rule passed but Tile/ScatterND cost too high |
| exp026_synthesis_orchestrator_core | orchestrator_ready | queue ready, base 6479.40 | - | phase-aware synthesis queue and worker task queue generated |
| exp027_cost_aware_lowering_bench | cost_guardrails_ready | 24563 candidate rows audited | - | cost-aware pre-emission guardrails generated |
| exp028_lookup_to_rule_miner | no_gain | 6479.40, +0.00 | - | lookup compression rule mining produced no pass candidates |
| exp029_crop_object_synthesizer | no_gain | 6479.40, +0.00 | - | constant Slice/Slice+Conv crop candidates did not improve |
| exp030_same_shape_transform_sweep | no_gain | 6479.40, +0.00 | - | cheap global transform/color-map/identity candidates did not fit |
| exp031_zero_component_fill_miner | rule_found | task187 rule pass, +0.00 | - | zero-component outside/inside fill explains task187; ONNX not generated |
| exp032_task187_component_lowering | improved_below_target | 6479.41, +0.0086 | - | task187 redundant And graph surgery improved cost 105313 -> 104413 |
| exp033_redundant_and_surgery_sweep | improved_below_target | 6479.52, +0.1097 | - | redundant And graph surgery improved 8 tasks |
| exp034_redundant_logic_surgery_sweep | improved_below_target | 6479.58, +0.0674 | - | redundant Or/comparison graph surgery improved 6 tasks |
| exp035_greedy_logic_surgery_composition | improved_below_target | 6479.66, +0.0782 | - | greedy graph-surgery composition accepted 16 steps across 7 tasks |
| exp036_noop_bypass_and_prune | diagnostic_high_risk | 6480.20 sample20, +0.5326 | - | broad no-op bypass improved 16 tasks but 4 failed full arc-gen |
| exp037_fullarc_filter_exp036 | improved_below_target | 6480.10, +0.4324 | - | full-arc filtered exp036 kept 12 robust task improvements |
| exp038_fullarc_gated_noop_bypass | improved_below_target | 6480.17, +0.0759 | - | full-arc gated greedy bypass improved 4 tasks |
| exp039_deep_fullarc_safe_bypass | improved_below_target | 6480.24, +0.0643 | - | deep full-arc safe-op bypass improved 4 tasks |
| exp040_deeper_fullarc_safe_bypass | improved_below_target | 6480.29, +0.0509 | - | deeper full-arc safe-op bypass improved 4 tasks |
| exp041_task145_deeper_mul_chain | submitted_lb_gap | 6480.30, +0.0164 | 3417.71 | task145-only full-arc Mul-chain bypass improved; user-requested submit ref 53414511 exposed large LB/local gap |
| exp042_freeop_dag_search_top200 | proof_positive | proxy +16.23 across 9 full-arc exact programs | - | DSL/DAG synthesis shows much larger upside than continued surgery; ONNX lowering pending |
| exp043_fixed_dsl_hit_lowering | no_gain | 6480.30, +0.00 | - | fixed rot/crop/upscale DSL lowering lost to existing artifacts |
| exp044_task031_dynamic_bbox_lowering | no_gain | 6480.30, +0.00 | - | bbox rule passed but full-grid GatherND cost was prohibitive |
| exp047_five_experiment_review_041_045 | review_complete | mixed but useful | - | 5-experiment PDCA review; course corrected away from lookup local score |
| exp048_submit_safe_seed_inventory | inventory_ready | strict seed 6282.23; teacher +197.17 local | - | exp005 submit-safe seed and P0/P1 target inventory generated |
| exp049_task020_teacher_profile | diagnostic_complete | strict cost 90133 vs teacher 3931 | - | task020 teacher is compact signature lookup; use as oracle only |
| exp050_task020_rule_probe | rule_probe_complete | 266 same-shape sparse fill examples | - | task020 is 3-cell background fill with preserved bbox |
| exp051_task020_orbit_completion_miner | partial | 96/266 pass | - | train orbit templates too narrow |
| exp052_task020_d4_orbit_rule | partial | 160/266 pass | - | D4 orbit rule passes test but misses 106 arc-gen examples |
| exp053_all_task_cost_250_600_inventory | inventory_ready | strict 6282.23; <=600 floor 7503.63; <=250 floor 7833.83 | - | all400 cost target master queue; 600だけでは7700不足、250寄りが必要 |
| exp054_signature_lookup_family_taxonomy | taxonomy_ready | 196 signature tasks classified | - | L4 62, L2 65, L3 31, L5 20, L1 8; first wave L1/L2 sparse fill |
| exp055_l1_sparse_fill_rule_miner | no_full_hit | 240 rules; best 132/266 | - | simple low-cost L1 rules insufficient; add object-role predicates |
| exp056_submission_calibration_inventory | inventory_ready | 67 experiments; 29 zips; submit-ready 0 | exp005 5929.89; exp041 3417.71 | next Kaggle submit should be first official-valid single-task delta |
| exp057_l1_object_role_feature_miner | no_full_hit | task020; 10 feature sets; 0 saved evals | - | AND object-role predicates too brittle |
| exp058_task020_residual_template_audit | audit_ready | 3 canonical templates | - | task020 reduces to 3-template selector problem |
| exp059_task020_three_template_selector | no_full_hit | best touch_rule 142/266 | - | next split template class and D4 orientation |
| exp060_task020_orientation_audit | audit_ready | true class oracle 266/266 | - | orientation unique; class selector remains |
| exp061_task020_class_feature_audit | audit_ready | canon_pos 24 keys 266/266 diagnostic | - | compress 24 keys into explainable selector |
| exp062_task020_canon_pos_rule_compressor | rule_found | nonlookup rule 266/266 | - | next tiny ONNX lowering and single-task LB calibration |
| exp063_task020_teacher_delta_calibration | rejected_validation_failed | 24_pass_1_fail | - | teacher single-task delta rejected; no submit |
| exp064_task020_rule_lowering_spec | lowering_spec_ready | 5-block ONNX plan | - | next correctness-first ONNX for task020 explicit rule |
| exp065_task020_lowering_reference | rule_reference_pass | reference 266/266 | - | input-only reference fixed; next ONNX cost<90133 then single-task LB calibration |
| exp066_task020_correctness_onnx_lowering | submitted_complete | strict local 6282.44, +0.2097 | 5930.10 | task020 explicit rule ONNX cost 90133->73080; LB delta matches local |
| exp067_public_notebook_utilization_audit | audit_ready | 400-task surgery/description audit | - | public notebooks converted into Seddik surgery queue and Karnak compiler priors |
| exp068_seddik_style_strict_scalarization | submitted_complete | strict local 6282.61, +0.1704 | 5930.27 | 34 full-arc-safe scalarization/dedup/prune improvements; LB delta matches local |
| exp069_karnak_prior_compiler_queue | queue_ready | 400-task compiler queue | - | Karnak prior selects LOCAL_PREDICATE_FILL and CROP_SHAPE as top compiler lanes |
| exp070_local_predicate_fill_feature_profile | profile_ready | top10 all complex sparse/object edit | - | simple template fill is insufficient; move to component/color-role profiling |
| exp071_component_color_role_profile | profile_ready | priority tasks 85, 251 | - | LOCAL_PREDICATE_FILL top10 split into fill/erase/recolor families; next predicate-tree target selected |
| exp072_task251_predicate_probe | probe_ready | TP 2708 / FP 113 / FN 0 base predicate | - | task251 reduced to FP-pruning branch tree; no ONNX/submission yet |
| exp073_task251_closed_component_shape_audit | audit_ready | positive components 367; rectangles 241 | - | task251 is closed zero-component/border reachability, not pure rectangle mask |
| exp074_task251_reachability_depth_surgery | no_gain | 7 candidates rejected | - | shallower reachability fails full arc-gen; no submit candidate |
| exp045_rule_searcher_best_practices_7700 | proof_plan_ready | projection 7834.73 if all400 cost<=250 | - | family best practices and phase projection prove 7700 as rule-searcher target |
| exp_b001_rule_replacement_backlog | backlog_ready | 400-task rule replacement backlog | pending | starts b-series; calib_001 strict seed submit ref 53414978 |
| exp_b002_p0_explainable_rule_sweep | no_full_hit | 0 full hits / 120 evals | - | queue-top P0 simple geometry failed |
| exp_b003_teacher_gain_p0_rule_sweep | no_full_hit | 0 full hits / 54 evals | - | teacher-gain P0 target corrected; task020 best partial 132/266 |
| exp_b004_teacher_gain_p0_structural_taxonomy | taxonomy_ready | 18 P0 tasks classified | - | next grammar is bbox-local/object-role |
| exp_b005_sparse_neighborhood_fill_sweep | no_full_hit | 0 train-fit candidates / 9 tasks | - | raw neighbor-count predicates are insufficient |
| exp_b006_five_experiment_review_b001_b005 | review_complete | review complete | - | b001-b005 useful as course correction; next exp_b007 object-role miner |
| exp_b007_bbox_local_role_miner | no_full_hit | 0 train-fit candidates / 9 tasks | - | fixed bbox-local coordinate/color-role rules insufficient |
| exp_b008_shape_conditioned_sparse_fill | no_full_hit | 0 full hits / 21 evals | - | shape branch table overfits train and fails arc-gen |
| exp_b009_bbox_affine_formula_miner | no_full_hit | 0 candidates / 9 tasks | - | coordinate formula grammar insufficient; move to object/component roles |
| exp_b010_component_object_role_sparse_fill_miner | rule_found | task037 full pass 266/266 | - | diagonal same-color ray fill explains task037 |
| exp_b011_task037_diag_ray_lowering | no_gain | 266/266 pass, cost 441976 vs 63726 | - | full-grid Conv visibility too expensive |
| exp_b012_task037_teacher_delta_calibration | rejected_validation_failed | 24_pass_1_fail | - | teacher delta not submit-safe |
| exp_b013_five_experiment_review_b007_b011 | review_complete | review complete | - | b007-b011 meaningful but not score-producing; next low-cost lowering |
| exp_b014_task037_fullarc_graph_surgery | no_gain | 22 candidates; 0 improved | - | strict artifact simple surgery cannot reduce task037 cost |
| exp_b015_l1_l2_sparse_fill_rule_bank_sweep | no_full_hit | 0 full hits / 260 evals | - | high-gain L1/L2 needs local predicate decision trees |
| exp_b016_l2_local_predicate_decision_tree_miner | no_full_hit | 0 train-fit / 185 screened | - | single local predicates fail top5 L2 train screen |
| exp_b017_l2_changed_cell_feature_profile | profile_ready | top5 L2 feature profile | - | ray features show high-precision positive islands |
| exp_b018_high_precision_island_or_tree_miner | no_full_hit | 0 train-fit / 4 trees | - | high-precision OR islands have too little coverage |
| exp_b019_five_experiment_review_b014_b018 | review_complete | review complete | - | b014-b018 meaningful but too narrow for score |
| exp_b020_l4_shape_crop_pattern_profiler | profile_ready | 0 lower candidates / 10 tasks | - | L4 top10 are not simple fixed-anchor/bbox crops |
| exp_b021_strict_seed_exp038_micro_delta_submit | submitted_complete | 6282.432095, +0.201867 | 5930.02 | strict seed + exp038 4task micro delta submitted ref 53417203 |
| exp_b022_public_notebook_intelligence | intelligence_ready | 7 notebooks classified | - | user-provided public notebooks mapped to source/lowering/taxonomy actions |
| exp_b023_seddik_surgery_pattern_audit | intelligence_ready | pattern audit ready | - | extracted uniform initializer / dedupe / pruning surgery patterns |
| exp_b024_safe_uniform_initializer_scalarization | submitted_complete | 6282.257713, +0.027485 | 5929.92 | strict-seed safe uniform initializer scalarization submitted ref 53417752 |
| exp_b025_submit_safe_delta_union | submitted_complete | 6282.812218 | 5930.40 | exp068 plus exp_b021 submit-safe delta union submitted ref 53418067 |
| exp_b026_five_experiment_review_b020_b024 | review_complete | review complete | - | b020-b024 useful but postpass-limited; return to rule/compiler replacement |
| exp_b027_task251_closed_zero_component_rule | rule_found | task251 rule 266/266 | - | closed zero-component fill rule found; lowering pending |
| exp_b028_task251_closed_component_lowering_probe | no_cost_gain | 266/266 but cost 138631+ | - | existing flood-fill lowering too expensive for task251 |
| exp_b029_task251_reachability_depth_surgery | no_gain | 0 valid / 7 candidates | - | task251 existing reachability depth not prunable |
| exp_b030_five_experiment_review_b025_b029 | review_complete | review complete | - | useful but lowering gap remains; next pivot to cheap compiler lanes |
| exp_b031_task085_horizontal_bar_alternate_erase_rule | rule_found | task085 rule 265/265 | - | horizontal bar alternate erase rule found; lowering pending |
| exp_b032_task085_artifact_surgery_probe | no_gain | 0 valid / 2 candidates | - | task085 Cast bypass invalid; artifact compact |
| exp_b033_low_cost_artifact_profile | profile_ready | cost<=250 19; cost<=600 27 | - | low-cost Gather/Slice/Pad/small Conv patterns profiled for compiler templates |
