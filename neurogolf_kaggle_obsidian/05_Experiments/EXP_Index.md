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
| exp075_task071_erase_rule_probe | no_simple_erase_rule | best 1/265 | - | task071 is not task085-style erase; route to recolor/copy/component branch-tree compiler |
| exp076_task071_recolor_copy_profile | profile_ready | out_color purity 0.9192; zero-mask purity 0.9399 | - | local features are informative but too many-key/lookup-like; split mask and copy direction before lowering |
| exp077_task366_object_anchor_crop_profile | profile_ready | exact crop 0/266 | - | task366 is not exact crop/object bbox/color-remap crop; move to panel/marker rule |
| exp078_task366_axis_halving_probe | no_simple_axis_halving_rule | best 0/266 | - | adjacent row/col pair reduction is not enough |
| exp079_task366_panel_overlay_probe | no_simple_panel_overlay_rule | best 0/266 | - | simple two-panel overlay is not enough |
| exp080_task366_marker_object_copy_rule | rule_found | 266/266 | - | source object panel + target marker panel copy rule solved; ONNX lowering pending |
| exp081_task366_lowering_inventory | inventory_ready | 695 templates; max used objects 3 | - | template enumeration too broad; lower structurally by small object/marker count |
| exp082_task366_onnx_primitive_cost_probe | cost_probe_ready | Slice+Pad 18021; marker mask 8115; Where 18001 | - | full-grid primitives too expensive for 600級; require sparse/small-patch lowering |
| exp083_sparse_update_primitive_cost_probe | cost_probe_ready | min sparse cost 10418; ScatterND 36036+ | - | small ScatterND/Gather still too expensive; task366 600級 lowering unresolved |
| exp084_task365_object_crop_rule_probe | rule_found | 266/266 | - | task365 solved by max color-2-count object crop |
| exp085_task365_lowering_inventory | inventory_ready | dense rectangles 266/266; max objects 3 | - | task365 is promising rectangle-window selector lowering candidate |
| exp086_task365_cost_minimization_html_probe | cost_probe_ready | padless cost 12; 3x3 Slice+Pad 381; 6x6 Slice+Pad 1461 | - | HTML FREE-op minimization helps, but task365 full single-model cost<=600 is unlikely with 6x6 padded output |
| exp087_small_output_crop_candidate_scan | candidate_scan_ready | P0 cropish 43 | - | max output area<=14 gives a stronger 600級 crop lane; top target task185 |
| exp088_task185_grid_2x2_compress_rule | rule_found | 267/267 | - | task185 solved by colored 4x4 lattice -> homogeneous 2x2 block compression |
| exp089_task185_homogeneous2x2_onnx_cost_probe | cost_probe_ready | core proxy cost 1147 | - | grouped Conv equality is above 600; try non-Conv equality or artifact surgery |
| exp090_task185_nonconv_2x2_cost_probe | cost_probe_ready | direct Mul proxy 2200; static lattice Mul proxy 2848 | - | non-Conv equality is worse; pivot task185 to artifact surgery/fused representation |
| exp091_task048_bridge_connectivity_rule | rule_found | 270/270 | - | task048 solved by color-8 component bridging both color-2 components |
| exp092_task048_small_connectivity_cost_probe | cost_probe_ready | onecell floor 52; 4step 7214; 8step 10542 | - | naive 8x8 connectivity unroll too expensive for 600級 |
| exp093_template_zip_official_compatibility | compatibility_audit_ready | official-compatible 0/7 | - | zip templates are uint8/int-grid dynamic-shape code, not drop-in official submissions |
| exp094_template_zip_official_reimplementation_cost | cost_probe_ready | one-node official ops cost 0-10; rot90 two-node 36004 | - | use zip as taxonomy for fused/one-node transforms only |
| exp095_one_node_template_rule_scan | scan_ready | 19 full hits | - | Python-grid one-node hits found, but padding semantics need official validation |
| exp096_one_node_template_replacements | no_gain | 0 accepted / 4 evaluated | - | full-grid one-node flips fail on padded one-hot; shape-aware transform needed |
| exp097_full30_one_node_template_scan | scan_ready | full30 tasks 3; hits 0 | - | full-grid one-node templates are not a broad score lane |
| exp098_shape_aware_template_replacements | no_gain | accepted 0/5; local delta 0.0 | - | shape-aware fixed 3x3 rot180 validates but costs 740 vs current 368; variable-shape flips need branches |
| exp099_oss_optimizer_stack_probe | tooling_probe_ready | local delta 0.0 | - | generic optimizer route not main lane; proceed to NeuroGolf-specific compiler |
| exp100_low_cost_artifact_rule_mining | compiler_catalog_ready | local delta 0.0 | - | low-cost artifacts converted into compiler archetypes/rewrite candidates |
| exp101_archetype_compiler_v1 | no_gain | 0 accepted / 9 generated | - | simple channel_gather/static_slice_pad already covered or loses cost |
| exp102_one_node_conv_kernel_mining | conv_kernel_catalog_ready | local delta 0.0 | - | one-node Conv local detector is promising for tiny local-mask tasks; ray visibility guarded |
| exp103_task185_conv_guided_surgery_review | no_gain | task185 cost 59584 unchanged; #1-#5 delta 0.0 | - | 5-experiment review: useful directionally, but next block must emit candidates |
| exp104_p0_fixed_3x3_static_index_probe | no_gain | fit 0 / 37 targets; local delta 0.0 | - | fixed absolute static index map does not fit P0 small-output tasks; move to anchor/shape-index compiler |
| exp105_bbox_anchor_onecell_slice_probe | no_gain | fit 0 / 4 onecell targets; local delta 0.0 | - | nonzero bbox-min anchor is too coarse for 1x1 P0 tasks |
| exp106_color_anchor_onecell_slice_probe | no_gain | fit 0 / 4 onecell targets; local delta 0.0 | - | color bbox-min anchor also fails; route 1x1 tasks to rule-specific diagnostics |
| exp107_task185_logic_bypass_surgery | no_gain | 39 full-pass / 240 candidates; local delta 0.0 | - | task185 local logic bypass validates but does not reduce official cost |
| exp108_compiler_campaign_10_review | review_complete | #1-#10 local delta 0.0 | - | not LB-effective yet; switch #11-#15 to task-specific fused compiler |
| exp109_task048_artifact_surgery_sweep | improved_bundle_candidate | local +0.023085; task048 cost 5609->5481 | - | first campaign local improvement after #10 reset; focused Cast bypass surgery |
| exp110_focused_surgery_rulehit_cropish_sweep | improved_bundle_candidate | local +0.086887; accepted 6 tasks | - | focused surgery generalizes; submit-safe post-pass/calibration lane |
| exp111_focused_surgery_second_pass | improved_bundle_candidate | local +0.013967; accepted 5 tasks | - | focused surgery compounds but sharply diminishes; keep as post-pass, pivot back to fused lowering |
| exp112_task185_intermediate_output_extraction | no_gain | output-shape intermediates 0; local delta 0.0 | - | task185 cannot be reduced by suffix/intermediate output extraction |
| exp113_compiler_campaign_15_review | review_complete | #11-#15 local +0.123939 | - | focused surgery useful for LB calibration but insufficient; next fresh fused lowering |
| exp114_task185_fresh_fused_lowering_probe | cost_probe_ready | best proxy cost 1516; validation 0_pass_1_fail | - | task185 core can be cheap-ish, but fixed lattice positions are wrong |
| exp115_task185_lattice_position_inventory | inventory_ready | 46 position patterns; 3 spacing patterns | - | task185 needs dynamic grid-line spacing/offset compiler, not raw position table |
| exp116_task185_window_selector_rule | rule_found | 267/267 | - | task185 lattice selector solved by 4-consecutive grid-line window scoring |
| exp117_task185_window_scoring_cost_probe | cost_probe_ready | selector proxies cost 4204 / 5160 / 76194 | - | direct ONNX window scoring is too expensive; review/pivot at #20 |
| exp118_compiler_campaign_20_review | review_complete | #1-#20 local +0.123939; #16-#20 local 0.0 | - | task185 diagnostics useful but no local gain; pivot #21-#25 to task366/task365 |
| exp119_task365_intermediate_output_extraction | no_gain | final-shape candidates 4; local delta 0.0 | - | task365 cannot be improved by simple suffix extraction |
| exp120_task366_intermediate_output_extraction | no_gain | final-shape candidates 6; local delta 0.0 | - | task366 final-shape candidates are mask/type tensors, not direct outputs |
| exp121_task366_cast_suffix_extraction | no_gain | 6 candidates; all 0_pass_1_fail | - | task366 final-shape masks are not semantically direct outputs |
| exp122_focused_surgery_third_pass_limited | improved_bundle_candidate | local +0.014053; accepted 5 tasks | - | focused surgery remains useful as post-pass, not main 7500 lane |
| exp123_compiler_campaign_25_review | review_complete | #21-#25 local +0.014053 | - | artifact harvesting failed; pivot #26-#30 to low-cost primitive compiler |
| exp124_focused_surgery_fourth_pass_limited | improved_bundle_candidate | local +0.004776; accepted 4 tasks | - | focused surgery tail continues but diminishes |
| exp125_focused_surgery_fifth_pass_limited | improved_bundle_candidate | local +0.004548; accepted 3 tasks | - | post-pass tail only |
| exp126_focused_surgery_sixth_pass_limited | improved_bundle_candidate | local +0.002847; accepted 2 tasks | - | tail converges to task263/316 |
| exp127_focused_surgery_seventh_pass_limited | improved_bundle_candidate | local +0.002854; accepted 2 tasks | - | final micro bundle candidate |
| exp128_compiler_campaign_30_final_review | final_review_complete | final local 6282.965236; total +0.153018 | - | LB7500 not reached; campaign stopped at #30 |
| exp129_compiler_farm_core | pipeline_core_ready | 6285 floor not proven; farm guardrails implemented | - | NeuroGolf IR/cost extractor/public CODE registry added; public CODE remains teacher/intelligence until LB evidence |
| exp130_public_code_6285_floor | rejected_lb_collapse | publicScore 1400.37 ref 53450559 | 1400.37 | beicicc golf-domain public CODE floor failed; do not adopt |
| exp131_farm_os_cost_probe | cost_probe_complete | 6 probes; proxy aligns on memory-heavy and Slice+Pad cases | - | cost extractor revised to numeric params+intermediate-memory proxy; official score_network remains source of truth |
| exp132_farm_step1_ir_emit_score | step1_complete | 7 real-task IR->ONNX->evaluate rows; 0 accepted | - | staged farm integration Step 1 complete; all controls failed validation as expected |
| exp133_farm_step2_step3_validation_ledger | pipeline_safe_operational | full-arc/ledger/safe bundle gate connected; accepted 0 | - | Step 2/3 complete; no submission.zip created because all controls failed sample validation |
| exp135_exp127_submission_confirmation | phase0_complete | exp127 ref 53480507 LB 5930.55; local/LB micro delta transferred | 5930.55 | 2026-06-10 plan Phase 0 confirmed; no duplicate submission |
| exp136_private_failure_subset_inventory | failure_subset_inventory_ready | 50 subset candidates; best 24 tasks sum to gap 352.41 exactly | - | Phase A private failure audit queue created; indirect evidence only |
| exp137_private_failure_consensus_audit | consensus_audit_ready | top consensus tasks: 013/002/029/009 all freq 20/20 | - | Use as private failure provenance/repair queue before bisection probes |
| exp138_priority_task_alternative_source_audit | alternative_audit_ready | 203 manifest rows, no improved alternative for 013/002/029/009/024/018 | - | Move to bisection probe for top4 consensus tasks |
| exp139_top4_failure_bisection_probe | probe_complete | ref 53520609 LB 5876.49; drop 54.06 matches all-alive expectation | 5876.49 | top4 013/002/029/009 are public-scoring alive; exclude from immediate failure suspects |
| exp140_next4_failure_bisection_probe | probe_complete | ref 53520700 LB 5886.89; missing drop identifies task018 public-zero | 5886.89 | task018 is first concrete +13.35 LB repair target |
| exp141_task018_teacher_repair_probe | blocked_validation | exp023 task018 teacher cost 17467 but 24_pass_1_fail | - | do not submit |
| exp142_task018_candidate_validation_audit | candidate_audit_complete | 6 unique raws; full-ok: strict, beicicc, exp_b035 | - | existing-source repair narrowed |
| exp143_task018_beicicc_repair_probe | no_public_gain | ref 53520866 LB 5930.55; beicicc also public-zero | 5930.55 | do not adopt |
| exp144_task018_b035_repair_probe | public_lb_improved | ref 53520958 LB 5942.48; task018 repair +11.93 | 5942.48 | new public LB best; high source/private robustness risk |
| exp145_next4b_failure_bisection_probe | probe_complete_all_alive | ref 53521106 LB 5859.97; tasks 032/050/016/031 all alive | 5859.97 | exclude from public-zero suspects |
| exp146_next4c_failure_bisection_probe | probe_complete_partial_zero | ref 53521220 LB 5887.02; missing drop identifies task025 public-zero | 5887.02 | task025 is next repair target |
| exp147_task025_candidate_validation_audit | candidate_audit_complete | 5 unique raws; only current raw full-ok and it is public-zero | - | task025 needs rule repair, not source swap |
| exp148_next4d_failure_bisection_probe | probe_complete_all_alive | ref 53521453 LB 5877.68; drop matches all-alive expectation for 366/077/064/084 | 5877.68 | exclude these tasks from immediate public-zero suspects |
| exp149_next4e_failure_bisection_probe | probe_complete_all_alive | ref 53521652 LB 5856.74; drop matches all-alive expectation for 053/056/067/005 | 5856.74 | exclude these tasks from immediate public-zero suspects |
| exp150_next4f_failure_bisection_probe | probe_complete_partial_zero | ref 53521805 LB 5890.05; missing drop identifies task023 public-zero | 5890.05 | task023 is next repair target |
| exp151_task023_candidate_validation_audit | candidate_audit_complete | 5 unique raws; current full-ok plus exp_b035 alternate full-ok | - | use exp_b035 task023 raw for repair probe |
| exp152_task023_b035_repair_probe | public_lb_improved | ref 53521952 LB 5956.28; task023 repair +13.80 over exp144 | 5956.28 | new current public LB best; high source/private risk |
| exp153_next4g_failure_bisection_probe | probe_complete_all_alive | ref 53522084 LB 5873.53; drop matches all-alive expectation for 051/049/046/011 | 5873.53 | exclude these tasks from immediate public-zero suspects |
| exp154_next4h_failure_bisection_probe | probe_complete_all_alive | ref 53522207 LB 5859.07; drop matches all-alive expectation for 033/087/276/030 | 5859.07 | exclude these tasks from immediate public-zero suspects |
| exp155_next4i_failure_bisection_probe | probe_complete_all_alive | ref 53522390 LB 5873.60; drop matches all-alive expectation for 063/022/037/020 | 5873.60 | exclude these tasks from immediate public-zero suspects |
| exp156_next4j_failure_bisection_probe | probe_complete_all_alive | ref 53522565 LB 5873.06; drop matches all-alive expectation for 027/082/088/092 | 5873.06 | exclude these tasks; exp157 prepared while pending |
| exp157_next4k_failure_bisection_probe | probe_complete_all_alive | ref 53522643 LB 5873.63; drop matches all-alive expectation for 090/173/028/091 | 5873.63 | exclude these tasks; consider task025 repair or dtype/cost lane |
| exp158_dtype_memory_cost_probe | cost_probe_complete | fp32 full-grid 36000, fp16 18000, bool/uint8 9000 | - | dtype width reflected; pursue targeted mask/dtype post-pass |
| exp159_dtype_postpass_candidate_audit | candidate_audit_complete | top dtype candidates: task366/284/158/382/187 | - | inspect task158 Cast/consumer pattern first |
| exp160_task158_dtype_rewrite_inspection | inspection_complete | 18 boolish-to-FLOAT casts all feed Sum | - | task158 direct dtype rewrite not safe; inspect Where/mask candidates next |
| exp161_where_dtype_rewrite_inspection | inspection_complete | Where-heavy candidates mostly already narrow dtype; removable-looking casts are final output casts | - | use dtype lane in new lowering design, not broad existing post-pass |
| exp162_task025_rule_diagnostic | diagnostic_complete | 266 examples; simple rules 0/266 | - | mine task025 changed-cell predicate next |
| exp163_task025_motion_predicate_mining | diagnostic_complete | axis-aligned motion; count preserved 64/266 | - | guide-line projection hypothesis |
| exp164_task025_line_projection_rule | rule_found | Python rule 266/266 | - | lower task025 guide-line projection to ONNX next |
| exp165_task025_line_projection_onnx_probe | no_cost_gain | 266/266 pass; cost 491600 vs 89286 | - | correctness confirmed; lower-cost representation needed |
| exp166_task025_line_projection_publiczero_repair | public_lb_improved | ref 53523413; expected 5968.17 | 5968.18 | new current public LB best |
| exp167_next4l_failure_bisection_probe | probe_complete_all_alive | ref 53523591; all-alive expected 5906.36 | 5906.33 | exclude 048/035/012/017 from public-zero suspects |
| exp168_next4l_candidate_validation_audit | candidate_audit_complete | full_ok_total 9 for 048/035/012/017 | - | held; exp167 all-alive |
