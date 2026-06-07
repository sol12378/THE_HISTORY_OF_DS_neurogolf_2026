# compiler_campaign_30

## Objective

革新的compiler設計に基づき、costを引き下げ、LB 7500到達を狙う。30実験を上限とし、5実験ごとにLB向上への有意義性をレビューし、10実験ごとにlocal estimate変化を測定して次実験へ反映する。

## Baseline

- current submit-safe local estimate: `6282.812218`
- current LB: `5930.40`
- calibration gap for submit-safe line: approximately `-352.34`

## Experiment Ledger

| Campaign # | Exp | Purpose | Local Delta | New Local | Decision |
|---:|---|---|---:|---:|---|
| 1 | exp099_oss_optimizer_stack_probe | 汎用OSS/ORT optimizerを主戦力にできるか確認 | 0.000000 | 6282.812218 | 汎用optimizer tuningではなく専用compilerへ進む |
| 2 | exp100_low_cost_artifact_rule_mining | 低cost artifactをcompiler archetype/rewrite候補へ変換 | 0.000000 | 6282.812218 | static_slice_pad/channel_gather/one_node_convを次の生成器にする |
| 3 | exp101_archetype_compiler_v1 | channel_gather/static_slice_pad/crop+colormapを全taskへ適用 | 0.000000 | 6282.812218 | acceptedなし。単純archetypeは現行bestに吸収済み |
| 4 | exp102_one_node_conv_kernel_mining | one-node Conv artifactを逆解析しConv compiler guardrailを作成 | 0.000000 | 6282.812218 | task185型小local-maskへone Conv primitiveを使う |
| 5 | exp103_task185_conv_guided_surgery_review | task185 one-Conv方針をsurgery/inventoryで評価し5実験レビュー | 0.000000 | 6282.812218 | accepted []; nextはcomputed position/index compiler |
| 6 | exp104_p0_fixed_3x3_static_index_probe | P0小出力taskへ固定9点static index map候補を試す | 0.000000 | 6282.812218 | 37対象でfit 0。絶対座標Gatherではなくanchor/shape indexへ進む |
| 7 | exp105_bbox_anchor_onecell_slice_probe | 1x1 P0 taskへnonzero bbox-min dynamic Slice候補を試す | 0.000000 | 6282.812218 | 4対象でfit 0。全nonzero bboxでは粗すぎる |
| 8 | exp106_color_anchor_onecell_slice_probe | 1x1 P0 taskへ色別bbox-min dynamic Slice候補を試す | 0.000000 | 6282.812218 | 4対象でfit 0。1x1 laneは一般anchorでなくtask別rule診断へ |
| 9 | exp107_task185_logic_bypass_surgery | task185既存artifactのlogic bypass surgeryを試す | 0.000000 | 6282.812218 | 240候補中39 full-passだが全てcost 59584でno gain |
| 10 | exp108_compiler_campaign_10_review | #1〜#10のlocal estimate変化測定と戦略レビュー | 0.000000 | 6282.812218 | local delta 0。#11〜#15はtask-specific fused compilerへ切替 |
| 11 | exp109_task048_artifact_surgery_sweep | task048既存artifactのfocused bypass surgery | 0.023085 | 6282.835303 | task048 Cast bypass full-pass; cost 5609->5481 |
| 12 | exp110_focused_surgery_rulehit_cropish_sweep | P0 cropish/rule-hit taskへfocused bypass surgeryを横展開 | 0.086887 | 6282.922190 | accepted 6 tasks: 184,187,207,263,316,394 |
| 13 | exp111_focused_surgery_second_pass | exp109/110 bundle上でfocused surgeryを2周目に進める | 0.013967 | 6282.936157 | accepted 5 tasks; compoundingするが明確に逓減 |
| 14 | exp112_task185_intermediate_output_extraction | task185既存artifactの中間output/subgraph extractionを試す | 0.000000 | 6282.936157 | output-shape intermediate 0。suffix cutでは削れない |
| 15 | exp113_compiler_campaign_15_review | #11〜#15のLB有意義性レビュー | 0.000000 | 6282.936157 | focused surgeryは有効だが浅い。次はfresh fused loweringへ |
| 16 | exp114_task185_fresh_fused_lowering_probe | task185 fresh fused loweringのcost floorとvalidationを測る | 0.000000 | 6282.936157 | best cost 1516だが固定latticeは0_pass_1_fail |
| 17 | exp115_task185_lattice_position_inventory | task185 lattice位置pattern/spacingをinventory化 | 0.000000 | 6282.936157 | 46 position patterns; spacingは3/4/5の3種類 |
| 18 | exp116_task185_window_selector_rule | 4連続grid-line window scoringでtask185 latticeを選ぶ | 0.000000 | 6282.936157 | Python selector rule 267/267。raw table不要 |
| 19 | exp117_task185_window_scoring_cost_probe | task185 window scoring/selectionのONNX cost floorを測る | 0.000000 | 6282.936157 | GatherND/ArgMax scoring 76194で重い。dynamic selectorは非主力 |
| 20 | exp118_compiler_campaign_20_review | #1〜#20 local estimate測定と#16〜#20レビュー | 0.000000 | 6282.936157 | #16〜#20 delta 0。task185 direct dynamic ONNXはpivot |
| 21 | exp119_task365_intermediate_output_extraction | task365既存artifactの中間output/suffix extractionを試す | 0.000000 | 6282.936157 | final-shape候補4件。改善なし |
| 22 | exp120_task366_intermediate_output_extraction | task366既存artifactの中間output/suffix extractionを試す | 0.000000 | 6282.936157 | final-shape候補6件はtype mismatch。改善なし |
| 23 | exp121_task366_cast_suffix_extraction | task366 final-shape mask候補にoutput dtype Cast suffixを付ける | 0.000000 | 6282.936157 | 静的検証は通るが全候補0_pass_1_fail |
| 24 | exp122_focused_surgery_third_pass_limited | accepted task限定でfocused surgery三巡目を試す | 0.014053 | 6282.950210 | accepted 5 tasks; post-passとしては継続価値あり |
| 25 | exp123_compiler_campaign_25_review | #21〜#25のLB有意義性レビュー | 0.000000 | 6282.950210 | task365/366 artifact harvestingは不発。#26〜#30は低cost primitive compilerへ |
| 26 | exp124_focused_surgery_fourth_pass_limited | accepted task限定でfocused surgery四巡目を試す | 0.004776 | 6282.954986 | accepted 4 tasks; tailはさらに逓減 |
| 27 | exp125_focused_surgery_fifth_pass_limited | accepted task限定でfocused surgery五巡目を試す | 0.004548 | 6282.959534 | accepted 3 tasks; post-pass tailのみ |
| 28 | exp126_focused_surgery_sixth_pass_limited | accepted task限定でfocused surgery六巡目を試す | 0.002847 | 6282.962381 | accepted 2 tasks; task263/316に収束 |
| 29 | exp127_focused_surgery_seventh_pass_limited | accepted task限定でfocused surgery七巡目を試す | 0.002854 | 6282.965236 | accepted 2 tasks; final submit-safe micro bundle候補 |
| 30 | exp128_compiler_campaign_30_final_review | 30実験campaign最終レビューと停止 | 0.000000 | 6282.965236 | LB7500未達。local +0.153018で停止 |

## Review Gates

- After #5: complete
- After #10: complete; local delta #1-#10 = `0.000000`
- After #15: complete; local delta #11-#15 = `0.123939`
- After #20: complete; local delta #1-#20 = `0.123939`
- After #25: complete; local delta #21-#25 = `0.014053`
- After #30: complete final stop; local delta #1-#30 = `0.153018`

## Review After #5

- verdict: mixed_but_directionally_useful
- local delta #1-#5: `0.000000` (all campaign deltas currently 0 unless exp103 accepted)
- analysis: No local/LB improvement yet, but the five experiments moved from generic optimizer hopes to a NeuroGolf-specific compiler grammar and ruled out simple archetype scans. This is useful only if the next block emits candidates from computed_slice_pad/one-Conv patterns rather than continuing catalogs.
- next policy: The next 5 experiments must include at least two score-producing candidate emissions, not only mining. Prioritize computed_slice_pad/tiny_dynamic_shape_index and task185/task087-style cost floor reductions.

## Review After #10

- verdict: not_lb_effective_yet_but_useful_for_strategy_reset
- local delta #1-#10: `0.000000`
- local delta #6-#10: `0.000000`
- analysis: The first 10 campaign experiments did not improve local estimate at all. Their value is negative evidence: generic OSS optimization, simple archetype rollout, absolute/static index maps, naive bbox/color anchors, and local task185 guard bypass are not sufficient. Continuing the same broad scans would be poor LB strategy.
- next policy: For #11-#15, require task-specific fused compiler work. Each experiment should target a solved or nearly-solved high-gain task and either emit a fused low-cost candidate or measure a concrete fused/subgraph cost floor. Priority lanes: task185 lattice homogeneous-2x2 fused lowering, task048 closed-form bridge feature/surgery, and low-cost artifact subgraph extraction. Avoid new broad anchor scans unless they are tied to a specific visual rule.

## Review After #15

- verdict: lb_useful_but_not_sufficient
- local delta #11-#15: `+0.123939`
- local delta #1-#15: `+0.123939`
- analysis: #11〜#13のfocused artifact surgeryは7 unique tasksでsubmit-safe localを押し上げ、LB較正候補として有意義だった。一方、#13で明確な逓減が出て、#14ではtask185のsuffix/subgraph extraction候補が0だった。これは既存artifact削りだけでは7500へ桁が足りないことを示す。
- next policy: #16〜#20はfresh fused loweringを必須にする。優先は task185 lattice homogeneous-2x2 の1-Conv以下/Pad最小lowering、task366 object-marker copyのfull-grid回避、task365 rectangle selectorのshape-branch低cost化。focused surgeryはbundle後post-passとして維持するが、主実験にはしない。

## Review After #20

- verdict: strategically_useful_but_no_local_gain
- local delta #1-#20: `+0.123939`
- local delta #16-#20: `0.000000`
- analysis: #16〜#20はlocalを増やさなかったが、task185について重要な判断を与えた。coreはcost `1516` まで下がる一方、固定座標はfailし、window selectorはPythonで `267/267` でもONNX selector plumbingが高costだった。したがってtask185のdirect dynamic ONNXは主力から下げる。
- next policy: #21〜#25はscore-producing候補を優先してtask366/task365へpivotする。task185はfused selectorの新案が出た場合のみ戻る。#25では5実験レビューを行い、local deltaが出なければ提出候補作成より別familyへ再pivotする。

## Review After #25

- verdict: artifact_harvesting_failed_but_postpass_still_useful
- local delta #21-#25: `+0.014053`
- local delta #1-#25: `+0.137992`
- analysis: #21〜#23のtask365/task366 suffix/artifact harvestingはlocalを全く動かさなかった。早期final-shape tensorは最終outputの意味を保持しておらず、Cast追加でも `0_pass_1_fail`。一方、#24のfocused surgery三巡目はaccepted 5 taskで微小deltaを出し、このlaneがsubmit-safe post-passとしては残ることを示した。
- next policy: #26〜#30ではtask365/task366のsuffix extractionを停止する。低cost artifactから逆輸入したone-node/fused data movement、小さい `Slice/Gather/Pad`、small local-mask `Conv`、safe post-pass surgeryを組み合わせたcost-aware compiler実験に戻す。full-grid `Where/GatherND/ScatterND` やdynamic `MatMul` は事前rejectする。

## Review After #30

- verdict: lb7500_not_reached_compiler_campaign_useful_but_micro_delta_only
- local delta #26-#30: `+0.015026`
- local delta #1-#30: `+0.153018`
- final working local estimate: `6282.965236`
- analysis: #26〜#29はfocused surgery tailをさらに刈り、submit-safe localを微小に伸ばした。しかしacceptedはtask263/316中心へ収束し、LB 7500へ必要な桁とは全く違う。30実験全体でscore-producingだったのは安全なgraph surgeryのみで、fresh compiler loweringsはcost/selector/writebackの壁でlocalを増やせなかった。
- final policy: このcampaignは30実験に到達したため停止する。exp127 bundleはLB micro-calibration候補だが、LB7500候補ではない。再開するなら、ONNX emission前にNeuroGolf専用IRとcost extractorを持つcompilerを作り、one-node/fused data movement、小さい `Slice/Gather/Pad`、small local-mask `Conv` だけをemitする。
