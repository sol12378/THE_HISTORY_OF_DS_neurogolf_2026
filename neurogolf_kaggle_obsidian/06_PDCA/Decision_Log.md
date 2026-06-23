# Decision Log

| Date | Decision | Rationale | Impact |
|---|---|---|---|
| 2026-06-05 | 初回ONNX検証taskは `task087` にする。 | 公開例が3x3固定の180度回転で、static shapeの `Gather` 2段で表現できる。 | 最小の手書きONNX baselineとして以後の検証基準にした。 |
| 2026-06-05 | exp001はCPU ONNX検証を優先し、GPU学習は次フェーズ以降に回す。 | 初回提出計画はpipeline一周が目的で、ONNX Runtime CPU検証と公式cost確認で十分。 | `task087.onnx` と単一task `submission.zip` を作成し、Kaggle submit ref `53383536` / public LB `14.50` を記録した。 |
| 2026-06-05 | exp002ではstrict static filterを維持し、6500未達でもKaggle submitしない。 | 高スコア公開artifactは `Compress`, functions, dynamic shapeでrejectされるものが多く、許容範囲を確認しないまま提出するとrule違反/不安定化リスクがある。 | local estimate `6282.08` の暫定bundleを保存。次PDCAはofficial rules再確認またはtop expensive task rewrite。 |
| 2026-06-05 | CUDAを導入し、GPUはrouting/rankingに使う。 | PyTorch CUDA smoke testが通り、RTX 2080 SUPERで軽量NN学習が実行できた。提出ONNXの採用判定はCPU official-like validationに固定する。 | `exp010_cuda_gpu_setup` と `exp011_gpu_route_classifier` を追加。GPU学習は有効化済み。 |
| 2026-06-05 | exp012のlookup改善は提出候補にしない。 | `signature_scatter_lookup` はlocal estimateを `6296.30` まで上げたが、arc-gen labelを利用するためleakage/overfitting riskが高く、6500にも届かなかった。 | exp012はlocal上限探索として保存。次はgraph surgeryと小型CNN/conv templateのstrict採用へ進む。 |
| 2026-06-05 | exp016は提出せず、次PDCAはremaining high cost taskの個別graph surgeryへ進む。 | top400 campaignでlocal estimateは `6479.38` まで上がったが、6500まで `20.62` 不足。採用改善もhigh-riskな `signature_scatternd_lookup` 中心だった。 | `experiments/exp016_top100_rewrite_campaign/submission.zip` はlocal upper boundとして保存するがsubmit候補ではない。 |
| 2026-06-06 | 7600へ向けてprogram synthesis pipelineへ移行する。 | exp017のiterative neighbor-fillはsample20で改善ゼロ。exp018の棚卸しでは、現baseの196 taskがsignature lookup由来で、残りhigh-cost taskもregion/line/crop/point patternに偏っている。 | `src/neurogolf/synthesis/` と `experiments/exp018_neurogolf_dsl_core/` を追加。次PDCAは `boundary_flood_fill`, `rectangular_room_fill`, `point_to_line_pattern`, `object_anchor_crop`, `static_lookup_compression`。 |
| 2026-06-06 | 単純unrollの `boundary_flood_fill` は主戦術から外す。 | exp019で `task187` はvalidation passしたが、18 step costが `237631` でbaseline `105313` より悪い。Top80でも改善ゼロ。 | 次は低memory loweringが可能な `diagonal_shift_tile`, `periodic_shift_fill`, `ring_depth_remap` を優先する。 |
| 2026-06-06 | full-grid `Where` chain loweringも主戦術から外す。 | exp020で `task313` の周期補完は24/24 passしたが、costが `215479` でbaseline `83572` より悪い。 | 次は sparse `ScatterND`、既存artifact graph surgery、signature lookup compressionを優先する。 |
| 2026-06-06 | 7600向けpipelineはcost-aware program synthesisとして設計する。 | exp019/020/021で、正しいruleでもnaive unroll / full-grid `Where` / large-coordinate `ScatterND` loweringはbaseline costに勝てないと確認した。 | exp022で `synthesis_backlog.csv` を生成。次は `static_lookup_compression`, `object_anchor_crop`, `sparse_coordinate_program`, `region_partition_fill`, `cost_model` を優先する。 |
| 2026-06-06 | `signature_scatternd_lookup` の未使用initializerを標準生成から削除する。 | exp023で195個のlookup modelに未使用initializerがあり、pruneだけでlocal `+0.015738` を得た。 | 今後のcandidate生成では同じ無駄を出さない。best local upperは `6479.398825`。 |
| 2026-06-06 | task203の全grid dynamic color-map loweringは主戦術から外す。 | exp024でvalidationは通ったが、candidate cost `216708` がbaseline `88402` を大きく上回った。 | ring-only更新、artifact surgery、cost modelによる全grid MatMul rejectを次PDCAへ回す。 |
| 2026-06-06 | task221の `Tile + mask` / dynamic `ScatterND` loweringは主戦術から外す。 | exp025でruleはvalidationを通ったが、costが `141047` / `165105` となりbaseline `55869` に負けた。 | block-prefix familyは `Slice/Gather` loweringまたはartifact surgeryを優先し、cost modelでfull-grid Tileと大きなScatterNDを警戒する。 |
| 2026-06-06 | 7600向けの主戦略をphase-aware program synthesis orchestratorに固定する。 | exp026でexp023 baseの400 taskを統合queueへ変換し、Phase 1から順にworkerへ渡せる `synthesis_queue.csv` と `worker_task_queue.csv` を作成した。Ad hoc templateでは6500以降の伸びが足りない。 | 次の改善は `exp028_lookup_to_rule_miner` と `exp029_crop_object_synthesizer` に集中する。Main agentはorchestrator、workerは低reasoningで狭い候補生成を担当する。 |
| 2026-06-06 | ONNX生成前にcost-aware guardrailを必ず通す。 | exp027で24563 candidate rowsを監査し、validation passしてもfull-grid `Where`, `Tile`, dynamic `MatMul`, large-coordinate `ScatterND` はbaselineに負けることを確認した。 | 候補生成時は `expected_cost_upper_bound` を持ち、上限超過ならONNX emissionしない。採用候補は `Slice`, `Gather`, small `Conv`, initializer pruningを優先する。 |
| 2026-06-06 | crop/resizeのconstant `Slice` 探索は主戦略から下げる。 | exp029で85 task/170 candidatesを検証したが改善ゼロ。高cost taskはconstant cropに合わず、pass candidateはすでに低costなtaskでbaselineに負けた。 | 次PDCAはsame-shape global transformのgraph surgery、line/grid fillのsmall `Conv`/`Gather`、既存artifact pruningを優先する。 |
| 2026-06-06 | cheap global transform/color-map sweepは主戦略から下げる。 | exp030で140 task/420 candidatesを検証したが生成candidateゼロ。`task255` を含む上位same-shape候補は単純変換では説明できなかった。 | 次PDCAはline/grid fillまたはpoint-to-line patternへ移し、small `Conv`/`Gather` と明示的cost gateを使う。 |
| 2026-06-06 | task187は規則探索からONNX lowering問題へ移す。 | exp031でzero-component outside/inside fillがtrain/test/arc-gen sample20に完全一致した。一方、exp019のnaive flood-fill unrollはcost `237631+` でbaselineに負けた。 | task187の次PDCAはclosed-form prefix mask、artifact surgery、または別のcost-aware loweringに限定し、naive unrollは再利用しない。 |
| 2026-06-06 | task187の冗長mask graph surgeryを採用し、同型探索へ広げる。 | exp032で `safe_name_76 = And(safe_name_16, safe_name_75)` を `safe_name_75` に置換し、task187 costを `105313 -> 104413` に下げた。 | local estimateは `6479.407407`。6500未達のため提出なし。次は高cost artifact全体で同型冗長nodeを探索する。 |
| 2026-06-06 | redundant `And` graph surgeryを新bestに採用する。 | exp033でtop120 taskの326 candidatesを評価し、8 taskを改善。local estimateは `6479.517116` に更新した。 | 6500未達のため提出なし。次は `Or`, `Where`, comparison node、unused initializerへgraph surgery探索を拡張する。 |
| 2026-06-06 | redundant logic graph surgeryを新bestに採用する。 | exp034でtop160 taskの1234 candidatesを評価し、6 taskを改善。local estimateは `6479.584508` に更新した。 | 6500未達のため提出なし。次はgraph surgeryのsequence compositionとunused initializer pruningを試す。 |
| 2026-06-06 | greedy graph-surgery compositionを新bestに採用するが、提出はしない。 | exp035でtop120 taskの2668 candidatesを評価し、16 steps / 7 tasksを改善。base accountingもexp032/033/034 overlayに修正したためlocal estimateは `6479.662741` に更新した。 | 6500未達のため提出なし。次はexp035 bundleのunused initializer pruning、より広い代数的graph surgery、低cost line/grid synthesisを並行する。 |
| 2026-06-06 | broad no-op bypassはfull-arc gateなしではbest採用しない。 | exp036はsample20で `6480.195302` まで伸びたが、追加の全arc-gen検証でtask80/145/187/268がfailした。 | exp036 bundleは提出しない。今後の広いgraph surgeryはfull-arc acceptanceをloop内に入れる。 |
| 2026-06-06 | exp037 full-arc filtered bundleを新bestに採用するが、提出はしない。 | exp037でexp036の16改善から全arc-gen通過の12 taskだけを残し、local estimateは `6480.095137` に更新した。 | 6500未達のため提出なし。次はrejected bypass patternのguardrail化とfull-arc greedy searchを行う。 |
| 2026-06-06 | full-arc gated greedy bypassを新bestに採用するが、提出はしない。 | exp038でcandidate採用時点に全arc-gen gateを入れ、4 tasks / 12 stepsを追加改善した。local estimateは `6480.171004` に更新した。 | 6500未達のため提出なし。次はfullarc candidate auditからsafe patternを抽出して探索順序とguardrailを改善する。 |
| 2026-06-06 | deep full-arc safe-op bypassを新bestに採用するが、提出はしない。 | exp039で全arc-gen通過実績task/opに絞り、4 tasks / 32 stepsを追加改善した。local estimateは `6480.235262` に更新した。 | 6500未達のため提出なし。次はsafe-op deepening継続と他taskへのpattern一般化を行う。 |
| 2026-06-06 | deeper full-arc safe-op bypassを新bestに採用するが、提出はしない。 | exp040でexp039をbaseに最大16passまで深掘りし、4 tasks / 38 stepsを追加改善した。local estimateは `6480.286118` に更新した。 | 6500未達のため提出なし。次はtask62/145の深掘り、または`Mul`連鎖削除の一般化を行う。 |
| 2026-06-06 | task145特化のdeeper `Mul` chain bypassを新bestに採用するが、提出はしない。 | exp041でtask145だけを最大32passまで深掘りし、17 stepsを追加改善した。local estimateは `6480.302478` に更新した。 | 6500未達のため提出なし。次はtask62の残り探索、またはtask145の`Mul` chain patternを他taskへ一般化する。 |
| 2026-06-06 | 6500突破の主戦略をartifact surgeryからDSL/DAG新規合成へ移す。 | exp042で全400 taskを軽量DSL scanし、9件のtrain/test/all arc-gen完全一致programを発見。proxy deltaは `+16.2291` で、exp040の追加surgery `+0.0509` の約319倍だった。 | exp043では固定flip/rot/crop/upscale hitをONNX loweringして実costを測る。bbox_nonzero task031は動的bbox loweringとして後続の高優先候補にする。 |
| 2026-06-06 | DSL発見とONNX loweringを分離し、naive loweringを主戦術にしない。 | exp043で固定rot/crop/upscaleは既存artifactより高cost、exp044でtask031 bbox ruleは正しいがfull-grid `GatherND` costが `1,050,228` になった。 | 次はtask031などの既存低cost artifactをprofileし、実際に安いshape/object表現を抽出する。full-grid dynamic index方式はguardrail入り。 |
| 2026-06-06 | exp041をuser requestで提出し、以後の主目的をrule searcher証明に切り替える。 | exp041 local estimateは `6480.302478` だったが、Kaggle ref `53414511` のpublic LBは `3417.71` で大きく乖離した。lookup/artifact依存baseは最終戦略として信頼できない。 | exp045でall400 cost<=250なら `7834.734477` という7700 projectionを作成。次PDCAはscore huntingではなくfamily別best practice、reject lowering、acceptance gateを備えたrule searcher構築へ集中する。 |
| 2026-06-06 | exp_b系列はstrict seedからKaggle較正し、小さいdelta提出でlocal/LBを統一する。 | exp041のLB collapseにより、大きなlocal upper bundleを一気に提出しても学習量が低い。まずexp005 strict seedのLB基準を取り、その後single-task/family deltaでlocalとLBの対応を見る必要がある。 | exp_b001で400-task rule replacement backlogとsubmission calibration planを作成。`calib_001_strict_seed_resubmit` としてexp005を提出、ref `53414978`。 |
| 2026-06-06 | exp_b001-b005の5実験レビュー後、次PDCAをbbox-local/object-role grammarへ切り替える。 | exp_b002/b003でD4・rectangle・lineの単純ruleはfull hitなし。exp_b005でraw neighbor-countもtrain-fitなし。一方exp_b004でP0の多くがsparse color-role fillまたはshape/object cropと判明した。 | exp_b007ではtask020/126/251/90/37を優先し、target color roleとbbox-local target coordinatesを明示するfeature minerを作る。full pass後のみcost<=250 loweringとsingle-task delta submissionへ進む。 |
| 2026-06-06 | exp016-041系local estimateをsubmit-safe CVとして扱わない。 | user-submitted exp041はlocal `6480.302` に対してpublic LB `3417.71`。差分 `-3062.59` は、signature lookup/public artifact依存がhidden生成評価で崩れたことを示す。 | 今後の主評価はhidden生成に耐える明示rule/DSL loweringへ寄せる。lookup bundleはteacher/diagnosticとしてのみ使い、submit候補から外す。 |
| 2026-06-06 | exp005をsubmit-safe seed、exp023/041をteacher-onlyとして扱う。 | exp048でexp005は400/400 all arc-gen passのstrict seed `6282.23`、exp023 teacherは+197.17 localだがexp041 LB崩壊により提出安全性がないと整理した。 | 次の改善はP0 teacher-gain taskを明示ruleへ圧縮する。task020から開始する。 |
| 2026-06-06 | all-task cost targetは600以下だけでなく250寄りを狙う。 | exp053でstrict seedから全400 taskをcost<=600へfloorしてもprojected scoreは `7503.63` で7700に `196.37` 不足。一方、cost<=250 floorは `7833.83` で `+133.83` margin。 | `task_cost_targets.csv` をmaster queueにし、signature lookup/current、crop_resize、sparse/object、point-to-lineを250寄りのsubmit-safe DSL/DAG最小回路へ圧縮する。 |
| 2026-06-06 | signature lookup圧縮のfirst waveはL1/L2 sparse fillにする。 | exp054で196 signature taskを分類し、L4 shape/cropが最大gainだが過去のfull-grid bbox lowering失敗リスクが高い。一方、L1/L2 sparse fillはsmall Conv/local predicate/tiny maskで250〜600化できる可能性がある。 | exp055で単純ruleを試したがfull hitなし。次はobject-role predicateとlocal frame推定を追加してからONNX loweringする。 |
| 2026-06-06 | P0 sparse fillはcoordinate-firstではなくobject/component-role firstで探索する。 | exp_b007の固定bbox-local coordinate、exp_b008のshape-conditioned branch table、exp_b009のbbox affine formulaがいずれもfull hitなし。特にb008はtrainに合ってもarc-gen未知shapeで崩れ、b009は候補すら生成できなかった。 | 次PDCAはcomponent/object-role sparse fill minerへ移す。nonzero component、hole/border、endpoint、symmetry orbit、nearest color object、changed-color roleを組み合わせ、full pass後のみtiny coordinate/mask loweringとsingle-task delta submissionへ進む。 |
| 2026-06-06 | task037は説明可能ruleとして採用候補だが、full-grid Conv loweringは採用しない。 | exp_b010でtask037 `opposite_ray_diag_same_color` が266/266 full pass。一方exp_b011のdepthwise Conv visibility loweringはbest cost `441976` でstrict `63726` より悪い。 | task037の次PDCAはdiagonal-specific small mask、triangular Gather、または既存artifact surgeryで<=600/<=250を狙う。full-grid Conv visibilityはguardrail入り。 |
| 2026-06-06 | task037 teacher single-task deltaは提出しない。 | exp_b012でteacher artifactはcost `8379` だが、full arc-gen validationが `24_pass_1_fail` で落ちた。 | local/LB calibration提出はfull validationを通ったdeltaだけに限定する。teacher artifactはrule oracleとしても慎重に扱う。 |
| 2026-06-06 | task037の単純graph surgery路線を下げ、rule-specific loweringへ戻す。 | exp_b014でstrict artifact 27 nodeに対する22 bypass候補を全arc-gen評価したが、改善ゼロ。3候補は `266_pass_0_fail` でもcostが変わらなかった。 | 次はdiagonal-specific mask、triangular Gather、precomputed diagonal basisなど、b010 rule専用の低cost表現を試す。 |
| 2026-06-06 | 高gain L1/L2 sparse fillは単一object-role ruleではなくlocal predicate decision treeを作る。 | exp_b015でgain_to_250上位20 taskにb010 rule bankを横展開したが、260 candidatesでtrain-fit/full hitなし。 | 次PDCAは3x3/5x5近傍、color-role、component-roleから小さいpredicate treeを合成し、full pass後にsmall Conv/mask loweringする。 |
| 2026-06-06 | L2 top tasksは手書き単一predicateを増やすよりchanged-cell feature profilingへ進む。 | exp_b016でtop5 L2に37単一predicateをtrain-screenしたが、185 candidatesすべてtrain exact-fitなし。 | 次はpositive/negative cell datasetを作り、target color role、row/col distribution、近傍色組み合わせ、component idを分析してbranch/treeを合成する。 |
| 2026-06-06 | L2 branch synthesisは高precision feature islandのOR-treeから始める。 | exp_b017でtask133/173/285/286にprecision 1.0のray feature値が見つかったが、recallは低い。単一featureではなく複数branchのORが自然。 | 次PDCAはpositive-only/high-precision feature valuesを貪欲に追加し、train full fitしたtreeだけfull arc-gen評価する。 |
| 2026-06-06 | 次PDCAは一度L2診断から提出候補を作りやすいlaneへピボットする。 | exp_b018でprecision island OR-treeはfalse positive 0でもcoverage不足。exp_b014-b018の5実験レビューではscore delta 0.0、提出候補なしと判定した。 | L2 findingsは保存し、次はL4 shape/cropまたは低cost artifact profileなど、より単純なloweringでlocal/LB較正候補を狙う。 |
| 2026-06-06 | L4 top10の単純固定crop routeは下げる。 | exp_b020でtop10 L4に固定anchor crop、shape-anchor crop table、nonzero bbox cropを試したがlower candidate 0。 | L4を続けるならobject-anchor/shape-transformが必要。提出候補優先ならL3 object moveまたはfull-arc-safe graph surgery deltaへ寄せる。 |
| 2026-06-06 | local/LB較正のため、strict seed + full-arc-safe micro deltaを提出する。 | exp_b021でexp038由来の4 task deltaが全arc-gen passし、local +0.201867の小さい差分になった。大きなlocal upper bundleではなく小deltaならLB差の原因分析がしやすい。 | Kaggle ref 53417203を提出。LB結果を待って、graph surgery deltaがLBで維持されるか確認する。 |
| 2026-06-06 | full-arc-safe micro deltaはLBである程度維持されるため、今後も小delta較正を使う。 | exp_b021 ref 53417203 はLB 5930.02で、strict seed 5929.89から+0.13。local +0.201867より小さいが方向は一致した。 | 大きなlookup-heavy bundleではなく、strict seed + small verified deltaの提出でlocal/LB環境を揃える。 |
| 2026-06-06 | ユーザー指定Kaggle notebooksを構造化知識として使う。 | exp_b022で7 notebookをpullし、blend source, Conv lowering, ONNX surgery precision, task taxonomy, baseline sourceに分類した。 | 次PDCAではseddik surgery/precisionから安全なgraph surgery候補、massimiliano/vyankteshからcheap Conv/Slice/Gather lowering、karnakからtaxonomy改善を取り込む。 |
| 2026-06-06 | seddik notebook由来の次手はsafe uniform initializer scalarizationにする。 | exp_b023で `find_compressible_initializers` / `compress_uniform_initializers` が抽出され、uniform tensorをscalar化する低リスクparameter削減routeが見えた。ただしnotebook側にもskip例がある。 | strict seed全taskへ候補生成し、static check + full arc-gen validation + official-like scoreで通ったものだけmicro-delta候補にする。 |
| 2026-06-06 | 今後の改善候補は小deltaでも早めにKaggle提出してlocal/LBを較正する。 | exp056でKaggle CLIを確認し、strict seed系はLB `5929.89` が再現、一方exp041 local upperはLB `3417.71` に崩壊済み。現時点でsubmit-readyな新規安全deltaは0。 | 次にofficial-valid single-task deltaが出たら温存せず提出し、local/LB gapをLB_Trackingへ即記録する。no-hit診断やteacher lookup bundleは提出しない。 |
| 2026-06-06 | task020は3-template selector問題として扱う。 | exp057のAND object-role featureは失敗したが、exp058で全266例の出力変更templateがD4正規化で3種類だけと判明した。 | 次は入力geometry/color-roleから3 templateを選ぶselectorを探索する。full pass後のみtiny loweringし、single-task delta提出でlocal/LB較正する。 |
| 2026-06-06 | task020ではtemplate classとD4 orientationを分離して探索する。 | exp059のbest `touch_rule` は142/266まで改善したが、templateが合っていても向き復元に失敗する例が多い。 | 次は3 template classの選択と、入力色配置からのorientation選択を別々に学習/規則化する。 |
| 2026-06-06 | task020の残課題を24-case `canon_pos` selectorの圧縮に絞る。 | exp060でorientationはclassが分かれば全266例で一意、exp061で`canon_pos` 24 keysがclassを完全分離すると分かった。 | raw lookup tableではなく、24 caseをcorner/edge/inner orbitの包含・欠損・接触条件へ圧縮し、full pass後にtiny loweringとKaggle single-task delta提出へ進む。 |
| 2026-06-06 | task020の小decision ruleをlowering候補に昇格する。 | exp062で`corners_eq1 -> corners`, `edge_mid_eq1 -> edge_mid`, `has_inner -> inner_diag`, default corners が266/266 full pass。raw lookupではなくorbit group条件なので説明可能性がある。 | 次はONNX loweringとcost計測。official-validかつ低costならsingle-task deltaとしてKaggle提出し、local/LB較正を行う。 |
| 2026-06-06 | task020 teacher artifactはsingle-task calibrationにも使わない。 | exp063でexp005 strict seedにtask020だけexp023 teacherを差し替えたが、全arc-gen validationが`24_pass_1_fail`で失敗した。 | Kaggle提出なし。task020はteacherではなくexp062の明示ruleをloweringする。 |
| 2026-06-06 | task020 rule loweringはcorrectness-first ONNXから始める。 | exp064でdynamic bbox cropだけでもcostが重くなる可能性を整理した。いきなり250〜600を狙うと実装失敗しやすい。 | まず266/266 passかつcost<90133を狙い、その後bbox/writebackを削って250〜600へ近づける。 |
| 2026-06-06 | task020明示ruleはONNX loweringへ進める。 | exp065で入力のみのreferenceがtrain/test/all arc-gen `266/266` を達成した。初回の1件失敗はcenter-only singletonのtarget color誤認で、候補除外により非lookup性を保ったまま解消した。 | 次はcorrectness-first ONNX loweringを実装し、costがstrict `90133` 未満ならstrict seed single-task deltaとしてKaggle提出してlocal/LBを較正する。 |
| 2026-06-06 | exp066 task020明示rule deltaをKaggleへ提出し、明示rule検証環境を信頼する。 | exp066でtask020 explicit rule ONNXが `266_pass_0_fail`、cost `90133 -> 73080`、local `+0.209732` を達成した。Kaggle ref `53417088` はLB `5930.10` で、exp005 strict seed `5929.89` から `+0.21`。local deltaとLB deltaが一致した。 | 小さいが重要な較正成功。teacher/public lookupではなく、明示rule + official validation + strict seed deltaの経路を主戦略として継続する。 |
| 2026-06-06 | user提示の公開notebookは再blendではなく、surgery/compiler priorとして使う。 | exp067で確認したところ、`massimilianoghiotto`, `konbu17`, `vyankteshdwivedi`, `magmacot`, `needless090` 系はexp002に既に取り込まれていた。一方、Seddik notebookはinitializer外科圧縮、Karnak notebook/datasetはtask description分類として未活用価値があった。 | Seddik-style auditをstrict seed post-pass queueにし、Karnakの400 task descriptionをexp053/054のcompiler lane重み付けへjoinする。公開artifactはそのまま提出せず、明示ruleまたはofficial-valid deltaへ変換して使う。 |
| 2026-06-06 | Seddik-style post-passをsubmit-safe bundle標準処理にする。 | exp068で37 taskを対象にunused/dedup/uniform-scalar surgeryをfull-arc gated適用し、34 task改善、local `+0.170391`。Kaggle ref `53417598` はLB `5930.27` で、exp066から `+0.17` と一致した。 | 今後の明示rule/compiler bundleにもSeddik-style post-passを適用する。ただしgainは小さいため、7700主経路は引き続きcost 250〜600級の新規rule/compiler合成。 |
| 2026-06-06 | LOCAL_PREDICATE_FILL上位はtemplate selectorではなくcomponent/color-role profilerへ進む。 | exp069でKarnak priorをjoinするとLOCAL_PREDICATE_FILLが最大gain laneだったが、exp070で上位10 taskを診断すると全て `complex_sparse_or_object_edit`。task020型の少数template fillは見つからず、task251もtemplate数84だった。 | 次のrule compilerはconnected component、color role、border/hole、row/column/ray構造を特徴化し、changed-cell predicate treeを合成する。 |
| 2026-06-06 | exp_b024のstrict-seed scalarization micro-deltaをKaggle提出する。 | exp_b024でstrict seed上の8 taskがfull-arc-safeに改善し、local `+0.027485`。Kaggle ref `53417752` はLB `5929.92` で、strict seedから `+0.03` とlocal差分に一致した。 | 狭いSeddik-style post-passもLBで安定することを確認。今後はpost-passとして標準化しつつ、主力はtask rule/compiler replacementへ戻す。 |
| 2026-06-06 | 既にLB較正済みのsubmit-safe deltaはunionして提出する。 | exp_b025でexp068にexp_b021の4 task deltaを重ね、全overlay taskがfull-arc passした。localは `6282.610350 -> 6282.812218`、Kaggle ref `53418067` はLB `5930.40`。 | current submit-safe bestを更新。大きなrule/compiler gainを作る前でも、検証済み小deltaをbest bundleへ積む運用を続ける。 |
| 2026-06-06 | exp_b020-b024は有意義だがpost-passに偏っていたため、次PDCAはrule/compiler本体へ戻す。 | 5実験レビューで、b021/b024はlocal/LB較正に成功し、b022/b023はnotebook活用を具体化した。一方、cost<=250 task数は増えず、L4/L2 compilerはまだ未解決。 | Seddik-style post-passは標準化する。次はobject-anchor crop、L3 object move/erase、task020型sparse completion横展開など、250〜600級の新規rule/compiler候補へ集中する。 |
| 2026-06-06 | task251をclosed zero-component fill ruleとしてlowering候補へ昇格する。 | exp_b027でtask251が「borderへ接しない0-componentかつ境界色集合が{2}なら1で塗る」ruleで `266/266` passした。row/column approximationは `195/266` で不十分。 | 次はclosed-component maskを安くONNX化する。naive flood-fill unrollは避け、border reachabilityやrectangle-specific maskをcost gate付きで試す。 |
| 2026-06-06 | task251にもnaive flood-fill unrollは使わない。 | exp_b028で既存boundary flood-fill loweringはstep 8から `266/266` になったが、最小cost `138631` でbase `100580` より悪い。 | region-fill compilerはclosed-form/rectangle-specific loweringへ寄せる。正しいPython ruleを見つけても、unrollで提出候補にしない。 |
| 2026-06-06 | task251既存artifactのreachability depth pruningは下げる。 | exp_b029でfinal reachability tensorを1〜7段浅い中間tensorへ差し替えたが、全候補がarc-gen mismatchになった。 | 既存artifact surgeryで深さを削るrouteは不採用。task251はclosed-form/rectangle-specific loweringか、別lane探索へ移る。 |
| 2026-06-06 | exp_b025-b029後、task251はrule hitだがlowering未解決として扱う。 | 5実験レビューで、LB best更新とtask251 rule発見は有意義だが、cost<=250 implementationは増えていないと確認した。 | 次PDCAはtask251 rectangle-specific closed maskを短く試し、難しければL3 object move / object-anchor crop / low-cost artifact profileへ移る。 |
| 2026-06-06 | task085をhorizontal bar alternate erase ruleとしてlowering候補へ昇格する。 | exp_b031でtask085がheight-3 horizontal rectangleの中央行をrelative parityで消すruleとして `265/265` passした。 | task085は説明可能rule hit。ただしrelative parity loweringが必要で、既存artifactもcompactなので短期score化は慎重に扱う。 |
| 2026-06-06 | task085の単純Cast bypassは不採用。 | exp_b032でCast bypass候補2件がORT type errorになった。内部graphはfloat16を期待しており、Castは必要。 | task085 artifact surgeryは単純削除では進まない。より強いpatternか別laneへ移る。 |
| 2026-06-06 | LOCAL_PREDICATE_FILLの次ターゲットをtask251 fillまたはtask085 eraseに絞る。 | exp071で上位10 taskをcomponent/color-role profileした結果、単純fillはtask251のみ、erase/mask cleanはtask085のみ、5 taskはmulti-color recolor/copy系だった。 | 次PDCAはfill templateではなくchanged-cell predicate tree synthesisを行う。multi-color系はrecolor/copy compilerへ分岐し、full arc-gen pass後のみONNX loweringとLB較正へ進む。 |
| 2026-06-06 | task251はbase predicate + FP pruning branch treeとして進める。 | exp072で `inside bbox + ray4 + adjacent nonzero` がTP 2708/FN 0を達成し、残りはFP 113だけになった。一方、straight adjacency単純rejectはFN 702で失敗した。 | 次はraw signature tableを避け、ray distance/local component contextを使った小branch treeを合成する。full passまでONNX loweringや提出はしない。 |
| 2026-06-06 | task251の単純depth削減は打ち切り、次はtask085または別connectivity loweringへ移る。 | exp073でtask251はclosed zero-component/border reachability問題と判明し、exp074で既存artifactのreachability depthを浅くすると全候補がfull-arc failした。 | task251 ruleは保持するが提出候補なし。score-producing探索はtask085 erase/mask clean、またはborder reachabilityの別表現へ移す。 |
| 2026-06-06 | LB6200+ public notebooksはcurrent submit-safe local estimateへ直接加算しない。 | exp_b034でcurrent submit-safe local `6282.8122` / LB `5930.40` の較正gap `352.4122` を使うと、LB6200相当 local は `6552.4122`、current から必要deltaは `+269.6` と見積もった。 | public blend artifactsはteacher/intelligence扱い。explicit rule / full-arc-safe graph surgery / LB micro-calibrationを通したものだけbest bundleへ採用する。 |
| 2026-06-06 | 次のcompiler設計は既存低cost artifactの実例をtemplate dictionaryとして使う。 | exp_b033でcurrent submit-safe bundleをprofileし、cost<=250は19 task、cost<=600は27 taskのみと確認した。低cost成功例は `Gather`, `Slice`, `Pad`, small `Conv` が中心で、naive region unrollではない。 | object-anchor crop、shape-index、small local mask loweringを優先する。未解決connectivity ruleは保持するが、full-grid flood-fillやlarge index loweringには戻らない。 |
| 2026-06-07 | task071は単純erase loweringではなくrecolor/copy compilerへ回す。 | exp075でtask085型horizontal-bar eraseは `0/265`、best candidateでも `1/265`。exp076で局所feature purityはout-color `0.9192` / zero-mask `0.9399` まで上がったが `3194` keys でlookup-like。 | task071は直接ONNX化しない。zero maskとsource-color copy directionを分離し、branch圧縮した候補だけlowering対象にする。 |
| 2026-06-07 | task366をobject-marker copy compilerの最優先lowering候補にする。 | exp077-079でnaive crop、axis halving、simple panel overlayを否定し、exp080でsource object panel + target marker panel copy ruleが train/test/all arc-gen `266/266` を達成した。task366はcurrent cost `836880` でgain_to_600 `7.24` / gain_to_250 `8.12` の最大候補。 | 次はdynamic component unrollを避け、panel split・marker shape matching・pasteを小さな `Slice`/`Gather`/`Where` 表現へ落とす。 |
| 2026-06-07 | task366のloweringでstatic template列挙を採用しない。 | exp081でsource templateが695種類、used templateでも511種類あり、estimated kernel paramsは7014。これはcost<=250〜600級やhidden汎化の観点でlookup-likeになりやすい。一方でmax source objectsは3、max marker cellsは7。 | object/marker countの小ささを使う構造compilerを設計する。connected component unrollやtemplate enumerationには戻らない。 |
| 2026-06-07 | task366で30x30 full-grid ONNX primitiveを積むrouteを下げる。 | exp082でfull-grid `Slice+Pad` が `18021`、marker `Conv+Tile` が `8115`、full-grid `Where` が `18001`。これらを組み合わせるとstrict costは改善してもcost<=600級には届かない。 | task366はsmall patch / sparse coordinate loweringへ寄せる。correctness-first full-grid ONNXは600級目標とズレるため優先しない。 |
| 2026-06-07 | task366の標準sparse writeback loweringを保留する。 | exp083で7 update `ScatterND` でも `36036+`、最小 `Gather` proxyでも `10418`。小さいpatchでも固定30x30 outputへ戻す段階でcostが膨らむ。 | task366は `266/266` rule hitとして保持するが、600級score productionは別表現が必要。次PDCAは既存低cost artifactに近いlaneへ戻す。 |
| 2026-06-07 | 次のscore-producing crop候補をtask365にする。 | exp084でtask365は `max_count2` object crop ruleが `266/266`。exp085で全objectがdense rectangle、object数は2〜3、selected shapeは16種類。task366より既存低cost `Slice/Gather/Pad` patternに近い。 | rectangle-window selector ONNX cost proxyへ進む。dynamic component extractionではなく、dense rectangle/window selectionとしてloweringする。 |
| 2026-06-07 | task365 full single-modelのcost<=600期待を下げる。 | exp086でPadなし小出力はcost `12` だが公式validationはpadded one-hot tensorを直接比較する。6x6 `Slice+Pad` proxyだけで `1461` になり、object selection前に600を超える。 | task365は大幅削減候補として保持するが、250〜600級の即戦力は最大output area<=14のcrop task、または6x6 floorを避ける別表現を探す。 |
| 2026-06-07 | 250〜600級crop laneを小出力P0 taskへ移す。 | exp087でmax output area<=14のcropish taskが43件あり、task185など固定3x3出力候補が多数見つかった。task365よりarea floorが低い。 | task185/022/271/079/242などを優先してrule probeし、full pass後にcost proxyを測る。 |
| 2026-06-07 | task185をrule hitとして保持するが、Conv型loweringは下げる。 | exp088でtask185は267/267 full pass。exp089でcore `Slice + grouped 2x2 Conv + Equal + Cast` proxyがcost `1147` となり、dynamic extraction前に600を超えた。 | 次は非Conv 2x2同色判定、または既存artifact graph surgeryでtask185の59584を大きく削る。 |
| 2026-06-07 | task185の非Conv shifted-Slice/Mul loweringも下げる。 | exp090でdirect shifted Slice + Mul proxyはcost `2200`、static lattice + Mul proxyは`2848`。中間3x3 tensorの数が支配的で、grouped Conv `1147` より悪い。 | task185はrule hitとして保持し、次は既存artifact surgeryまたはfused表現探索へ移る。 |
| 2026-06-07 | task048を1x1 rule hitとして保持するが、naive connectivity loweringは下げる。 | exp091でbridge connectivity ruleが`270/270`。exp092で1x1 output floorは`52`だが、4step dilation `7214`、8step `10542`。 | 既存artifact surgeryまたはclosed-form path特徴を探す。connectivity unrollを標準routeにしない。 |
| 2026-06-07 | `neurogolf_templates.zip` はdrop-inではなくtaxonomyとして使う。 | exp093で代表7テンプレートの公式互換は`0/7`。zipは`uint8/int32 [H,W]`, `in/out`, dynamic shape前提で、公式は`FLOAT [1,10,30,30]` one-hot `input/output`を直接比較する。 | 直接提出しない。公式one-hotへ個別に再実装し、score_networkで再測定する。 |
| 2026-06-07 | zip由来テンプレートは「公式one-hotで1ノードにできるもの」だけ優先する。 | exp094でofficial flip/rot180/channel gatherはcost `4/8/10` と極小。一方、2ノードrot90は中間30x30により`36004`。 | 1ノード/fused data-movementの探索を優先し、30x30中間を作る複合テンプレートは事前rejectする。 |
| 2026-06-07 | Python-grid one-node hitsは公式padding semanticsで必ず再検証する。 | exp095で19 full hitsを見つけたが、exp096で公式one-hot replacement 4件が全て`0_pass_1_fail`。full-grid Sliceがpaddingを動かしてしまう。 | 可変サイズflip/rotはshape-aware crop/transform/padが必要。full-grid 1ノード変換はfull 30x30 active taskかpadding不変taskに限定する。 |
| 2026-06-07 | full-grid 1ノードtemplate routeを主戦力から外す。 | exp097で全例30x30 taskは3件だけで、1ノードgeometry/recolor hitは0。 | zip由来taxonomyは保持するが、次はshape-aware compiler、artifact surgery、または小出力closed-form ruleへ戻る。 |
| 2026-06-07 | shape-aware flip/rot template routeも近短期のscore laneから外す。 | exp098でtask087/140の固定3x3 rot180はfull validationを通ったがcost `740` で現行 `368` より悪い。task150/155は可変shape、task380はmismatch。 | local estimateは `6282.812218` のまま。zipはtaxonomyとして使い、250〜600級の主探索は小出力rule、artifact surgery、低cost artifact pattern miningへ戻す。 |
| 2026-06-07 | 30実験compiler campaignでは汎用ONNX optimizer tuningを主戦力にしない。 | exp099で主要OSS moduleは未導入、ORT generic optimizerはexp009でno gain、今回のoffline serialization probeでも不安定だった。汎用optimizerはNeuroGolfの `memory+params` costを直接最小化しない。 | 次は低cost artifact mining、NeuroGolf専用rewrite rule、cost-aware extraction/e-graph compilerへ進む。 |
| 2026-06-07 | compiler grammarは低cost artifactから逆輸入する。 | exp100でcost<=2000 artifact 83件を分類し、`one_node_conv_kernel` と `computed_slice_pad` が各20件で最大archetypeだった。 | 汎用DSL探索ではなく、`channel_gather`, `static_slice_pad`, `one_node_conv`, `computed_slice_pad`, `tiny_dynamic_shape_index` を最初のgrammarにする。 |
| 2026-06-07 | 単純channel_gather/static_slice_pad横展開は下げる。 | exp101で全task scanして9候補を生成したがaccepted 0。候補は現行bestと同等または高costだった。 | 次は単純archetypeではなく、one-node Conv local detectorとcomputed slice/index patternの中身を使う。 |
| 2026-06-07 | one-node Convは小local-mask primitiveとして使い、visibility/connectivity用途はguardrailで制限する。 | exp102で低cost Conv artifact 23件のうち16件が`sparse_local_pattern_detector`。一方、過去task037のray visibility Convはcost 441976で失敗済み。 | 次はtask185型の小lattice/local-maskへone Conv detectorを適用する。large ray/connectivity Convは事前rejectする。 |
| 2026-06-07 | compiler campaign #1-#5後、catalog-only実験を続けない。 | exp099-103のlocal deltaは0。方向修正としては有意義だが、LB向上には実candidate emissionが必要。task185単純surgeryもcost `59584` のまま。 | 次の5実験では `computed_slice_pad` / `tiny_dynamic_shape_index` から最低2本のscore-producing candidateをemitする。 |
| 2026-06-07 | 固定絶対座標のstatic Gather compilerをP0小出力の主戦力にしない。 | exp104で37 P0 small-output cropish taskをscanしたが、各出力セルが全例で同一入力座標から来るtaskは0。candidate生成もacceptedも0でlocal estimateは `6282.812218` のまま。 | `Gather` はprimitiveとして保持するが、次はobject/shape/lattice anchorからoffsetを計算する `computed_slice_pad` / `tiny_dynamic_shape_index` compilerへ進む。 |
| 2026-06-07 | 1x1 crop laneで全nonzero bbox-min anchorを主戦力にしない。 | exp105でtask355/346/48/291にnonzero bbox-min + fixed offsetを試したがfit 0。全nonzero bboxは出力セル選択のanchorとして粗すぎる。 | 次に1x1 laneを続ける場合は、色別bbox・component role・marker色anchorに限定する。fit後のみdynamic Slice候補をemitする。 |
| 2026-06-07 | 1x1 crop laneの一般anchor-offset scanをいったん止める。 | exp106で色別bbox-min + fixed offsetもtask355/346/48/291にfit 0。単純marker近傍抽出では説明できない。 | 1x1 taskはtask別rule診断へ戻す。campaign #9は3x3固定出力のshape-index/crop lane、またはtask185/task048のfused lowering/graph surgeryを優先する。 |
| 2026-06-07 | task185の局所logic bypass surgeryを主戦力にしない。 | exp107で240候補を評価し、39候補がfull validation passしたが全てcost `59584` のまま。局所guard削除は公式score上の改善にならない。 | task185は小手先のbypassではなく、4x4 lattice homogeneous-2x2 ruleのfused lowering、または大きなsubgraph extractionへ進める。 |
| 2026-06-07 | compiler campaign #11〜#15はtask-specific fused compilerへ切り替える。 | exp108で#1〜#10 local deltaを測定した結果、`0.000000`、accepted task count 0。広いanchor scanや局所bypassを続けてもLB向上に直結していない。 | task185 lattice homogeneous-2x2 fused lowering、task048 closed-form bridge/surgery、low-cost artifact subgraph extractionを優先する。広いscanは視覚ruleが先にある場合だけ行う。 |
| 2026-06-07 | rule-hit focused artifact surgeryを#11〜#15の有力laneにする。 | exp109でtask048 Cast bypassがfull validation `270_pass_0_fail`、cost `5609 -> 5481`、local `+0.023085` を達成した。#1〜#10のbroad scanではlocal 0だったため、方針転換直後の改善として意味がある。 | task048 accepted candidateはsingle-task LB calibration候補。次は同じfocused surgeryを他のrule-hit/low-cost-near taskへ広げるか、task048 delta unionを作る。 |
| 2026-06-07 | focused artifact surgeryをsubmit-safe post-pass laneとして保持する。 | exp110で35 targetへ横展開し、6 task accepted、local `+0.086887`。#11〜#12合計で `+0.109972` となり、#10以前のbroad scanより明確に有効。 | このlaneはLB calibration候補としてbundle化する。一方で7500には桁不足なので、#13以降はfused lowering/subgraph extractionで大きいcost削減も狙う。 |
| 2026-06-07 | focused artifact surgeryの深掘りはpost-pass扱いに留める。 | exp111の2周目は5 taskを追加改善したがlocal deltaは `+0.013967` まで逓減した。#11〜#13合計は `+0.123939` で、安全だが7500へ向かう主火力ではない。 | #14以降はこのbundleをsubmit-safe較正候補として保持しつつ、task185 lattice、task366 object-marker copy、task365 rectangle cropなどのfused lowering/subgraph extractionへ戻す。 |
| 2026-06-07 | task185は既存artifactのsuffix extractionではなくfresh fused loweringへ切り替える。 | exp112でtask185 artifactの241 intermediate tensorsを調べたが、final `1x10x30x30` output shapeの中間tensorは0だった。既存graphは最終outputを早期生成して後段で加工する構造ではない。 | task185の次手は4x4 lattice homogeneous-2x2 ruleを直接短いONNXへ落とすこと。局所bypass/suffix cutの優先度を下げる。 |
| 2026-06-07 | compiler campaign #16〜#20はfresh fused loweringを必須条件にする。 | #15 reviewで#11〜#15 local deltaは `+0.123939` と有意義だったが、focused surgeryは逓減し、7500には桁不足。 | task185/task366/task365など、既にPython ruleが解けている高gain taskを、full-grid中間を避ける専用compilerで低cost化する。 |
| 2026-06-07 | task185の次課題をcore loweringからgrid-line position detectionへ移す。 | exp114でfinal Pad込みcore proxyはcost `1516` まで下がったが、固定lattice位置は全variant `0_pass_1_fail`。exp115でraw位置は46 pattern、spacingは3/4/5の3種類と判明した。 | raw position tableを避け、spacing/offsetをinput geometryから推定するdynamic compilerを設計する。 |
| 2026-06-07 | task185はraw position tableではなくwindow scoring compilerとして進める。 | exp116で4連続grid-line row/col window pairを非background交点数でscoreするselectorが `267/267` を達成した。 | #19ではONNXでwindow scoring/selectionのcost floorを測る。raw coordinate table化は禁止し、input-only geometryとしてloweringする。 |
| 2026-06-07 | task185のdirect dynamic ONNX selectorを主戦力から下げる。 | exp117でline detectionはcost `4204`、branchless spacing coreは`5160`、GatherND/ArgMax window scoringは`76194`。Python selectorは正しいが、素直なONNX化は600級にも現行costにも勝ちにくい。 | #20 reviewでtask185継続はfused selectorがある場合に限定する。次のscore-producing候補としてtask366/task365へのpivotを検討する。 |
| 2026-06-07 | #21〜#25はtask185からtask366/task365へpivotする。 | #20 reviewで#16〜#20 local deltaは0。task185はcoreもselector ruleも分かったが、direct ONNX selectorが高cost。 | 次の5実験は、task366 object-marker copyまたはtask365 rectangle selectorなど、別のsolved high-gain ruleでscore-producing loweringを狙う。 |
| 2026-06-07 | task365の単純suffix extractionを主戦力にしない。 | exp119でtask365 final-shape intermediate候補4件を評価したが、3件はtype mismatch、1件はvalidation mismatchで改善なし。 | task365を続けるならfresh object selector/shape branch cost probeに限定する。score-producing優先ならtask366へ寄せる。 |
| 2026-06-07 | task366の単純suffix extractionを主戦力にしない。 | exp120でfinal-shape candidate 6件を評価したが、全てelem type mismatchで静的検証に失敗した。現行artifactは`1x1x30x30` maskを大量生成している。 | task366を続けるならmask-to-float suffix最小化かfresh representationに限定する。単純output extractionは打ち切る。 |
| 2026-06-07 | task366 artifact harvestingを打ち切り寄りにする。 | exp121でoutput dtype Cast suffixを足しても6候補すべて`0_pass_1_fail`。早期final-shape maskは最終outputに意味的に近くない。 | task366を続けるならfresh representationのみ。ただし過去cost probeが重いため、#24では別score-producing laneを探す。 |
| 2026-06-07 | focused surgeryはpost-passとして保持するが主戦力にはしない。 | exp122で三巡目もaccepted 5 task、local `+0.014053` を出した。一方、#11〜#13からの逓減は明確で、LB 1位に必要な桁ではない。 | exp122 bundleはLB calibration候補として保持し、#26〜#30の主実験は低cost primitive compilerへ移す。 |
| 2026-06-07 | task365/task366 suffix extraction laneを停止する。 | #21〜#23はすべてlocal delta 0。#25 reviewで、final-shape中間tensorは最終outputの意味を保持していないと判断した。 | #26〜#30はone-node/fused data movement、小さい `Slice/Gather/Pad`、small local-mask `Conv`、safe post-pass surgeryに集中する。 |
| 2026-06-07 | 30実験compiler campaignを停止する。 | exp128で#30に到達。final localは `6282.965236`、total deltaは `+0.153018`。LB7500は未達で、改善はgraph surgery micro-deltaに限定された。 | exp127はmicro-calibration候補として保持するが、これ以上このcampaignでは実験しない。再開するならONNX emission前のNeuroGolf専用IR/cost extractorを作る。 |
| 2026-06-07 | 6285 public CODE floorは証拠付きregistry gateを通すまで未達扱いにする。 | exp129で既存公開CODE/blend系をregistry化したが、6285+ Kaggle LB evidenceは見つからず、best observed localも `6276.160646`。公開artifactを直接下限扱いすると過去のpublic/LB崩壊やleakage riskを再発させる。 | public CODEはteacher/intelligenceとして取り込み、full-arc validation + cost extraction + LB calibrationを通過したものだけ6285 floorに昇格する。 |
| 2026-06-07 | exp_b037 beicicc golf blendを6285 floor候補として提出する。 | exp130でexp_b037 zipを固定化し、400 task / parse failure 0 / size violation 0 / banned op 0を確認。public referenceはbeicicc LB 6645だが、local official costはgolf domainで未確定。 | Kaggle ref `53450559` として提出。publicScoreが6285以上ならfloor registryを昇格し、未満ならpublic CODE floor仮説を棄却または再blendへ戻す。 |
| 2026-06-07 | exp130 public CODE floor仮説を棄却する。 | Kaggle ref `53450559` はpublicScore `1400.37`。sanityとtrain/test validationが通っても、golf-domain public artifactはこのworkspaceの提出LBへ転移しなかった。 | exp130はbest/floorに採用しない。current submit-safe bestはexp_b025 LB `5930.40`。public CODEはtask別teacher/intelligenceとして使い、直接提出下限にはしない。 |
| 2026-06-07 | farm cost extractorはノード数ではなく数値proxyとofficial scoreを中心にする。 | exp131でnode数ベースだと`small_where_expand`のofficial cost `36000`を軽く見積もる危険が確認された。一方、final output memoryを除外した中間memory proxyは`static_slice_pad_3x3`でproxy `381` / official `360`と近い。 | pre-emissionではparams+intermediate-memory proxyで候補を並べる。accepted判定は必ずofficial `score_network`、full-arc validation、bundle ledgerで行う。 |
| 2026-06-07 | farmは採用候補がない場合にsubmission.zipを作らない安全停止を標準動作にする。 | exp133でStep 2/3を接続し、7 control候補がsample validationで全滅したため、full-arcはnot_run、ledger全件rejected、bundle_created falseになった。 | 今後の候補生成もこのgateを通す。full-arc validかつcost改善した候補だけBundleLedger acceptedになり、accepted 0ならzipを作らない。 |
| 2026-06-10 | 2026-06-10計画のPhase 0は完了扱いにし、exp127を重複提出しない。 | Kaggle submissionsでexp127 ref `53480507` が既に完了しており、LB `5930.55` はlocal micro delta `+0.153` と整合した。gapは約 `-352.42` で維持された。 | current submit-safe bestを exp127 / exp135 confirmed に更新し、次PDCAはprivate failure task特定とdtype/cost probeへ進む。 |
| 2026-06-10 | private failure 特定は subset-sum 候補を監査queueとして使い、即bisection提出しない。 | exp136で24 taskのscore合計がgap `352.41` と完全一致する候補が得られたが、subset-sumは状況証拠であり直接証明ではない。 | 次は上位subset候補のconsensus/provenance/shape外挿リスクを監査し、修復可能taskまたはprobe groupを絞る。 |
| 2026-06-10 | private failure の初回個別監査queueを task013/002/029/009/024/018 にする。 | exp137でtask013/002/029/009が上位20 subsetすべてに出現し、exact-gap subset 4件にもすべて含まれた。task024/018も高頻度。 | 次PDCAはこのqueueの代替source探索・現モデル構造監査・shape外挿リスク確認を行い、修復候補またはbisection groupを作る。 |
| 2026-06-10 | task013/002/029/009をgap原因suspectから外す。 | exp139 fail-stub probe ref `53520609` はLB `5876.49` で、exp127からのdrop `54.06` が4 task all-alive期待値 `54.05675` と一致した。 | 次のprivate failure探索はこの4 taskを除外し、task024/018/004/019/032/050/016などの次点groupへ移る。 |
| 2026-06-10 | task018を最初のpublic-zero repair targetにする。 | exp140 fail-stub probe ref `53520700` はLB `5886.89` で、task024/018/004/019 all-alive期待drop `57.01524` に対してobserved dropは `43.66`。不足分はtask018 points `13.35099` と一致した。 | task018を修復できればLB `+13.35` 級を回収できる可能性があるため、次PDCAはtask018の代替source/rule/robust lowering探索に集中する。 |
| 2026-06-10 | exp144をcurrent public LB bestとして採用候補にするが、private-safeとは扱わない。 | exp144 ref `53520958` はtask018をexp_b035候補に差し替え、public LB `5942.48` を達成した。exp127比 `+11.93` で期待値と一致し、task018 public-zero修復に成功した。 | LB bestを更新。ただしexp_b035はpublic/source blend由来のため、最終safe候補化にはrisk reviewと追加repair検証が必要。 |
| 2026-06-10 | task032/050/016/031をpublic-zero suspectから外す。 | exp145 fail-stub probe ref `53521106` はLB `5859.97` で、exp127からのdrop `70.58` が4 task all-alive期待値 `70.57623` と一致した。 | 次のpublic-zero探索は既知alive集合を除外し、task021/014/025/008/366/077/064などへ進む。 |
| 2026-06-10 | task025を次のpublic-zero repair targetにする。 | exp146 fail-stub probe ref `53521220` はLB `5887.02` で、task021/014/025/008 all-alive期待drop `57.12815` に対してobserved dropは `43.53`。不足分はtask025 points `13.60040` と一致した。 | task025の既存source/rule repair候補を監査し、単独repair probeを作る。 |
| 2026-06-10 | task025は既存source差し替えではなくrule repair対象にする。 | exp147でtask025の34 source rows / 5 unique rawsを監査したが、full-okは現行 `franksunp_blended_best` rawのみで、これはexp146によりpublic-zeroと推定済み。 | 短期LB最大化では次のpublic-zero探索を継続し、task025は別途rule/robust lowering queueへ送る。 |
| 2026-06-10 | task366/077/064/084をpublic-zero suspectから外す。 | exp148 fail-stub probe ref `53521453` はLB `5877.68` で、exp127からのdrop `52.87` が4 task all-alive期待値 `52.78042` とほぼ一致した。 | 次のpublic-zero探索はこの4 taskを除外して未probe subset候補へ進む。task025はrule repair queueに保持する。 |
| 2026-06-10 | task053/056/067/005をpublic-zero suspectから外す。 | exp149 fail-stub probe ref `53521652` はLB `5856.74` で、exp127からのdrop `73.81` が4 task all-alive期待値 `73.81309` と一致した。 | 次のpublic-zero探索は残り頻出候補 `023/036/085/034/051/049/046` などへ進む。 |
| 2026-06-10 | task023を次のpublic-zero repair targetにする。 | exp150 fail-stub probe ref `53521805` はLB `5890.05` で、task023/036/085/034 all-alive期待drop `55.52734` に対してobserved dropは `40.50`。不足分はtask023 points `15.02731` と一致した。 | task023の既存source/rule repair候補を監査し、単独repair probeを作る。 |
| 2026-06-10 | exp152をcurrent public LB bestとして採用候補にするが、private-safeとは扱わない。 | exp151でtask023の別raw `exp_b035_new_source_full_arc_blend` がfull validation `266_pass_0_fail` / cost `72612` と判明し、exp152 ref `53521952` でpublic LB `5956.28` を達成した。 | LB bestを更新。ただしtask018/task023ともにpublic/source blend由来repairのため、final safe候補化にはrisk reviewと追加repair検証が必要。 |
| 2026-06-10 | task051/049/046/011をpublic-zero suspectから外す。 | exp153 fail-stub probe ref `53522084` はLB `5873.53` で、exp127からのdrop `57.02` が4 task all-alive期待値 `57.02309` と一致した。 | 次のpublic-zero探索は残り頻出候補 `033/087/276/030/063/022/037/020` などへ進む。 |
| 2026-06-10 | task033/087/276/030をpublic-zero suspectから外す。 | exp154 fail-stub probe ref `53522207` はLB `5859.07` で、exp127からのdrop `71.48` が4 task all-alive期待値 `71.47883` と一致した。 | 次のpublic-zero探索は残り頻出候補 `063/022/037/020/027/082/088/092` などへ進む。 |
| 2026-06-10 | task063/022/037/020をpublic-zero suspectから外す。 | exp155 fail-stub probe ref `53522390` はLB `5873.60` で、exp127からのdrop `56.95` が4 task all-alive期待値 `56.94773` と一致した。 | 次のpublic-zero探索は残り頻出候補 `027/082/088/092/090/173` などへ進む。 |
| 2026-06-10 | task027/082/088/092をpublic-zero suspectから外す。 | exp156 fail-stub probe ref `53522565` はLB `5873.06` で、exp127からのdrop `57.49` が4 task all-alive期待値 `57.48547` と一致した。 | 次のpublic-zero探索は、採点待ち中にzip準備済みの `090/173/028/091` へ進む。 |
| 2026-06-10 | task090/173/028/091をpublic-zero suspectから外し、Phase A bisection継続の優先度を下げる。 | exp157 fail-stub probe ref `53522643` はLB `5873.63` で、exp127からのdrop `56.92` が4 task all-alive期待値 `56.91703` と一致した。これで高頻度subset候補の多くはaliveと確認済み。 | 残り候補は薄い。追加bisectionは低優先で保持し、task025 rule repairやPhase C dtype/cost laneを次の主候補にする。 |
| 2026-06-10 | Phase C-1 dtype/cost laneをtargeted post-passとして進める。 | exp158で公式 `score_network` がdtype幅を反映することを確認。fp32 full-grid中間 `36000`、fp16 `18000`、bool/uint8 `9000`。一方、追加演算で中間が増えるとcostは戻る。 | broad fp16化ではなく、既存artifactのboolean/mask系full-grid中間を狙った局所post-passを次に試す。 |
| 2026-06-10 | dtype post-pass初回のgraph監査対象をtask158にする。 | exp159でtask366/284/158などが上位候補になったが、task158はboolish-to-FLOAT full-grid Castが18個あり、局所rewrite可能性を確認しやすい。 | task158のCast consumer patternを精査し、validかつcost改善がありそうなら単task candidateを作る。 |
| 2026-06-10 | task158の直接dtype rewriteを保留する。 | exp160でtask158の18個のfull-grid boolish-to-FLOAT Castがすべて`Sum`へ入る数値カウント用途と判明した。BOOLのままでは同じ意味を保ちにくい。 | 次はWhere条件やmask selection周辺のdtype candidateを優先してinspectする。 |
| 2026-06-10 | 既存artifactへのbroad dtype post-passを主戦力から下げる。 | exp161でWhere-heavy候補task206/338/328/366を監査したが、削れそうなCastは最終output Castか、既にFLOAT16/UINT8/BOOLを使ったbranchだった。 | dtype cost知見は保持するが、既存graph後処理ではなく新規lowering/DSL生成時に狭dtype中間を設計する。 |
| 2026-06-10 | task025 repairは単純ruleではなくchanged-cell predicate miningへ進める。 | exp162で266例を診断した結果、shape preservedかつ出力色は入力既存色だけだが、identity/fill/mirror/rot180系は全て`0/266`で、変更は純粋なzero-fillでもなかった。 | task025は引き続きpublic-zero repair targetだが、次PDCAは差分セルの生成・消去位置と色移動構造を特徴量化してrule候補を作る。 |
| 2026-06-10 | task025をguide-line projection ruleとしてlowering queueへ昇格する。 | exp163でaxis-aligned motionが濃厚になり、exp164のinput-only Python ruleが`266/266` full passした。完全な縦/横guide lineを持つ色のstray cellをline隣接セルへ射影し、guide lineなしの色を消す規則で説明できる。 | 次PDCAはtask025専用ONNX loweringとcost validation。current modelはpublic-zeroかつcost `89286` なので、full validation passなら単独repair submit候補になる。 |
| 2026-06-10 | exp165 task025 correctness-first loweringは採用しない。 | exp165は`266_pass_0_fail`で正しいが、candidate cost `491600` がbaseline `89286` を大きく上回った。9色分のfull-grid reduction/MatMul/mask展開がmemory costを膨らませた。 | task025 ruleは保持するが、このONNXは提出しない。次は色loop圧縮、guide色抽出、または既存artifact surgeryで同ruleを安く表現する。 |
| 2026-06-10 | exp166をcurrent public LB bestに更新する。 | exp165 candidateはlocal costではno gainだったが、task025 public-zero repairとしてKaggle ref `53523413` に提出したところpublic LB `5968.18` を達成し、期待値 `5968.17` と一致した。 | current public bestをexp166へ更新。task025はpublic-zero repair成功。ただしfinal private-safe候補化にはhidden robustness risk reviewが必要。 |
| 2026-06-10 | task048/035/012/017をpublic-zero suspectから外す。 | exp167 fail-stub probe ref `53523591` はpublic LB `5906.33` で、exp166からのdrop `61.85` が4 task all-alive期待drop `61.82369` と一致した。 | この4 taskは短期public-zero repair queueから除外。exp168でfull-ok既存候補は確認済みだが、即repairではなくcost/source分析用に保持する。 |
| 2026-06-10 | task133/task158をpublic-zero repair targetにする。 | exp169 fail-stub probe ref `53523802` はpublic LB `5938.07` で、all-alive expected `5912.36` より `25.71` 高かった。不足dropはtask133+task158の点数と一致し、task396/task047はaliveと解釈できる。 | task133/task158の既存source候補を優先監査し、distinct full-local-valid repair candidateがあれば即提出する。 |
| 2026-06-10 | exp172をcurrent public LB bestに更新する。 | exp170でtask133/task158のexp_b035 full-local-valid rawを確認し、exp172 ref `53523884` でpublic LB `5993.82` を達成した。期待値 `5993.8188` と一致し、exp166から `+25.64`。 | current public bestをexp172へ更新。ただしexp_b035 public/source blend由来のため、private-safeとは扱わず、次もpublic-zero探索とrisk reviewを継続する。 |
| 2026-06-10 | task003/038/001/086をpublic-zero suspectから外す。 | exp171 fail-stub probe ref `53524036` はpublic LB `5928.74` で、exp172からのdrop `65.08` が4 task all-alive期待drop `65.07777` と一致した。 | この4 taskは短期public-zero repair queueから除外する。 |
| 2026-06-10 | task285をpublic-zero repair targetにする。 | exp174 fail-stub probe ref `53524145` はpublic LB `5980.81` で、exp172からのdrop `13.01` がtask286 alive分と一致し、missing drop `13.06` がtask285点数に一致した。 | task285のexp_b035 full-local-valid rawをrepair probeとして提出する。task286は短期public-zero suspectから除外する。 |
| 2026-06-10 | exp178をcurrent public LB bestに更新する。 | exp178 ref `53524248` はtask285をexp_b035 full-local-valid rawに差し替え、public LB `6005.93` を達成した。期待値 `6005.9322` と一致し、exp172から `+12.11`。 | current public bestをexp178へ更新。6000 public LBを突破したが、exp_b035由来のためprivate-safeとは扱わず、risk reviewと追加探索を継続する。 |
| 2026-06-10 | risk-inventory上位16 taskをpublic-zero suspectから外す。 | exp176/179/181/182のfail-stub probesはそれぞれ task202/382/205/383, 251/109/239/358, 281/203/126/159, 313/370/234/303 を対象にし、全てpublic LBがall-alive期待値と一致した。 | これら16 taskは短期repair候補から除外。次はexp183 task187/204/198/364の採点結果を確認しつつ、未probe risk候補へ進む。 |
| 2026-06-10 | task187をpublic-zero repair targetにする。 | exp183 fail-stub probe ref `53526873` はpublic LB `5965.28` で、all-alive expected `5951.8651` より `+13.41` 高かった。不足dropはtask187 points `13.4353` と一致した。 | exp185で見つけたdistinct full-local-valid task187 rawを使い、exp186 repairを提出してLB回収を確認する。 |
| 2026-06-10 | exp186 task187 massimilianoghiotto repairを採用しない。 | exp186 ref `53526998` はpublic LB `6005.90` で、exp178 `6005.93` を更新しなかった。期待 `6019.37` と大きく乖離し、このfull-local-valid rawも公開repairにならない。 | current bestはexp178のまま。task187は未解決repair queueへ戻し、探索はexp184 risk probeへ進める。 |
| 2026-06-10 | task284/300/379/340をpublic-zero suspectから外す。 | exp184 ref `53527058` はpublic LB `5951.06` で、all-alive expected `5951.0653` と一致した。 | 4 taskを短期repair候補から除外し、次はexp188 task328/301/306/387 probeへ進む。 |
| 2026-06-10 | task328/301/306/387をpublic-zero suspectから外す。 | exp188 ref `53527133` はpublic LB `5950.27` で、all-alive expected `5950.2733` と一致した。 | 4 taskを短期repair候補から除外し、次はexp190 task238/112/377/177 probeへ進む。 |
| 2026-06-10 | task238/112/377/177をpublic-zero suspectから外す。 | exp190 ref `53527204` はpublic LB `5950.05` で、all-alive expected `5950.0478` と一致した。 | 4 taskを短期repair候補から除外。Phase Aの追加bisectionは限界効率が下がっているため、次はreview gateを挟んでPhase C cost-band圧縮へ移る。 |
| 2026-06-10 | Phase A lower-ranked bisectionを主戦場から下げる。 | exp192で直近8 probes / 32 tasksをレビューした結果、7 groupがall-alive、1 groupのtask187 suspectもexp186 repairでno gainだった。 | task187はfocused repair queueに残すが、主PDCAはdocs planどおりPhase C cost-band compressionへ移行する。 |
| 2026-06-10 | 最終output Cast除去だけのdtype post-passを採用しない。 | exp194でtask206/328のBOOL output版はfull validation passしたが、official costは変わらなかった。 | Phase C dtype圧縮は最終Castではなく、内部full-grid中間またはfresh lowering設計を対象にする。 |
| 2026-06-10 | C-2の汎用channel gather / fixed shift slice-padを主力から下げる。 | exp195はmapping_ok_count 2 / improved 0、exp196はshift_ok_count 1 / improved 0。どちらも提出可能なcost改善を出さなかった。 | 次のPhase C supplierは `one_node_conv_kernel`、またはtask085/185/048など解決済みruleの専用loweringへ移す。 |
| 2026-06-10 | 汎用1x1 Conv colormapを主力から下げる。 | exp198は非injective color mapも許した1x1 Convを全taskへ適用したが、mapping_ok_count 4 / improved 0だった。exp197の汎用3x3 fitは現実装が重すぎた。 | Phase Cはtask-specific solved-rule loweringへ移す。3x3 Convを続ける場合はvectorized minerとして別途作り直す。 |
# 2026-06-10 exp200/201: task185 axis-separable selectorを次のPhase C候補に戻す

- 背景: exp117のpairwise selector proxyはcost `76194` でbaseline `59584` を超え、task185 dynamic loweringを一度deprioritizeしていた。
- 新証拠: exp200でrow/col独立score selectorが `267/267` passし、pairwise selectorと完全一致。exp201でfused axis selector proxy cost `4156` を確認。
- 決定: task185はPhase C C-3の有望候補として復帰。次はcorrectness-first ONNX loweringを試す。
- リスク: selector proxyは正解artifactではない。spacing/core selectionと最終3x3出力の動的接続でcostが増える可能性がある。

# 2026-06-10 exp202: task185はdynamic-index candidateへ進める

- 背景: selectorが安くても、選択後の4x4 lattice extractionが高ければbaselineを超える懸念があった。
- 新証拠: `GatherElements` 2段 extraction + homogeneous core + Pad はcost `7660`。exp201 selector proxy `4156` と合わせたprojected costは `11816` でbaseline `59584` より十分低い。
- 決定: task185の次実験はdynamic selected row/col indexを作るcorrectness-first ONNX candidateにする。all-pair window列挙とstatic Slice分岐爆発は避ける。
- リスク: exp202はstatic index proxyであり、dynamic index生成とcolor/bg handlingをつなぐと追加costやruntime制約が出る可能性がある。

# 2026-06-10 exp203: task185 selector realism後も継続

- 背景: exp201 selector proxyはspacing 3/4/5の実構造より単純で、過小見積もりの可能性があった。
- 新証拠: spacing 3/4/5 dilated axis selector + ArgMax のcostは `5236`。exp202 extraction/core `7660` と合わせても `12868`。
- 決定: task185 loweringを継続し、次はdynamic index生成を実装する。cost budget上はbaseline `59584` に十分余裕がある。
- リスク: bg色検出、special-cell mask、ArgMax indexからGatherElements index tensorへの変換が未実装。

# 2026-06-10 exp204-206: task185初回dynamic candidateは不採用、debug継続

- 背景: exp203まででselector/extractionのcost proxyはbaseline内と見えたため、exp204で実際にdynamic candidateを接続した。
- 結果: runtime/staticは通るがvalidationは `0_pass_1_fail`。さらにfull index template initializerによりcostが `106720` 以上となりbaseline超過。
- 診断: exp205ではPython arbitrary selectorが `267/267` なので、任意start選択だけが原因ではない。exp206ではbasicがchannel0のみ、nonzero score variantが位置ズレ・色欠落を起こす。
- 決定: exp204 candidateは提出しない。次に進むなら、compact start/spacing tableでindex paramsを削減しつつ、selector indexとoutput 3x3位置の対応を小さなdebug graphで確認する。
- リスク: task185に時間を使いすぎる可能性がある。次の1-2実験でfull passの兆しが出なければ別C-3 taskへpivotする。

# 2026-06-10 exp207/208: task185にはdynamic bg detectorが必要

- 背景: exp204の位置ズレがselector index由来かcore由来か不明だった。
- 新証拠: exp207でONNX中間を確認すると、example 0でPython selector `[5,8,11,14]` に対し、ONNX nonzero-score variantは `[2,5,8,11]` を選んでいた。exp208で背景色は1-9全てに分散していると判明。
- 決定: 固定bg maskによるtask185修正は不可。継続するなら動的にmost-common nonzero/bg色を推定してselector/coreから除外する必要がある。
- リスク: dynamic bg detectorはcostと実装複雑度を増やす。task185は有望性を残すが、次は別C-3 taskへpivotして機会費用を下げる判断も妥当。

# 2026-06-10 exp209-214: task346をsolved-rule assetに追加

- 背景: task185/task365はdynamic selectionが重くなったため、1x1 cropish候補へpivotした。
- 新証拠: exp211でtask346 `least_nz` が `263/267`。exp213で4 switchを説明する構造特徴 `rank0_largest_component >= 8` を発見し、exp214で `267/267` full-pass。
- 決定: task346をC-3 solved-rule assetとして追加する。ただしONNX loweringはcount + component largest-size proxyが必要なので、提出候補ではない。
- リスク: threshold 8 は4例から導出されておりhidden過学習リスクがある。ONNX化前にcost proxyとより構造的な解釈が必要。

# 2026-06-10 exp215: task346 direct loweringは保留

- 背景: task346のfull-pass ruleはcountは簡単だがcomponent-size補正を含む。
- 新証拠: color count + ArgMinはcost `111` と十分安い一方、component growth proxyは1 stepでも `117135` とbaseline `9178` を大きく超える。
- 決定: task346は提出candidate化しない。component-freeな小出力taskやcount-only ruleの探索へ戻る。
- リスク: task346のcomponent補正を別表現で安く出せる可能性は残るが、標準full-grid reachabilityでは不可。

# 2026-06-10 exp216/217: 1x1単純aggregate laneを縮小する

- 背景: task346が `least_nz` で近かったため、exp087の全1x1 cropish候補へ単純aggregate ruleを広げた。
- 新証拠: exp216でfull hitはなく、task346のみ `interior_least_nz 265/267`。exp217で残り2 failはsimple count/edge/corner条件ではTP2/FP0に分離できなかった。
- 決定: 1x1単純aggregate laneは主力から下げる。task346はcomponent-like補正を安く表現できるまで保留。
- リスク: 2例補正を無理に足すとhidden過学習になりやすい。

# 2026-06-10 exp218-220: task039 bbox crop ruleは正しいが現行loweringでは不採用

- 背景: 3x3固定cropish候補の安いSlice/transform ruleを探索した。
- 新証拠: exp218でtask039が `bbox_top_left 3x3` により `264/264` full pass。exp219の`GatherElements` loweringはvalidation passだがcost `48219` でbaseline `7772` を超過。exp220のdynamic `Slice` はstatic checkerで `dynamic shape crop` reject。
- 決定: task039はsolved-rule assetとして保持するが、提出candidateにはしない。dynamic bbox cropはstatic-shape-preservingな低cost表現が見つかるまで保留。
- リスク: 正しいruleを高cost loweringで提出するとscoreが下がる。task135 full hitもbaseline `360` のため優先度は低い。

# 2026-06-10 exp221/222: fixed small-shape crop/colormap supplierを主力から下げる

- 背景: 3x3 cropではtask039が当たったがdynamic bbox cropが高costだったため、固定anchorの小shape hitを探した。
- 新証拠: exp221のfull hitはtask039/135/326のみ。fixed-anchor hitのtask135/326はbaseline costが `360` / `160` と低すぎる。exp222のglobal color-map追加でも新規有用hitはなく、full hitはidentity mapだけ。
- 決定: fixed small-shape crop/colormap supplierは高cost taskの主力から下げる。
- リスク: shape branchやmask抽出を含むfamilyは未探索なので、crop laneの失敗をsmall-output全体の失敗とは見なさない。

# 2026-06-10 exp223/224: task300をsolved-rule assetに追加する

- 背景: fixed crop/colormapでは高cost候補が出なかったため、小出力taskのshape/mask familyを棚卸しした。
- 新証拠: exp223でtask300が高costshape full-hit候補。exp224で出力が全例「最大size 4-connected same-color componentのbbox crop」と一致した。`rank_size_desc=267/267`、`crop_exact=267/267`。
- 決定: task300をsolved-rule assetへ追加し、次にONNX cost proxyを試す。
- リスク: 最大component選択とdynamic bbox cropは既存guardrail上高cost化しやすい。正しいruleでもcost wallを確認するまで提出候補にしない。

# 2026-06-10 exp225-234: task300 max-color ruleを採用しcurrent bestを更新する

- 背景: exp224の最大component crop ruleは正しいが、component growthは高cost化が懸念だった。
- 新証拠: exp225で最大nonzero色countが最大componentと全例一致し、component growth不要と分かった。exp233でchannel GatherElements版ONNXが `267_pass_0_fail`、cost `77546 -> 52653` を達成。
- 決定: exp234としてexp178 current bestにtask300だけ差し替え、Kaggleへsingle-task delta提出する。ref `53530035` はpublic LB `6006.32` で期待値と一致したため、current public bestをexp234へ更新する。
- リスク: ruleは全available examples由来なのでhidden edgeは残るが、入力のみ構造ruleでpublic source rawではない。現base exp178自体のprivate riskは継続。

# 2026-06-10 exp235: task174はtask300型の単純max-color cropでは解けない

- 背景: exp223でtask174も高cost shape-full-hit候補だったため、task300と同じmax-color crop laneを試した。
- 新証拠: max_colorはlargest componentと `266/266` 一致するが、max_color cropは `138/266` のみ。
- 決定: task174は即ONNX loweringしない。component内部のsub-crop/shape ruleを先に監査する。
- リスク: 高cost候補だが、無理にtask300 loweringを流用するとvalidation/costの両方で失敗する。

# 2026-06-10 exp236/237: task174/task253を一旦保留する

- 背景: task174はtask300に続く高costshape候補、task253は固定binary template候補だった。
- 新証拠: exp236でtask174の単純内部sub-cropはbest `138/266` のまま。exp237でtask253の出力templateは固定だが、色selectorはbest `76/265` と弱い。
- 決定: 両taskとも直ちにloweringしない。より説明力のあるrule minerを後で設計し、短期PDCAは別の高cost候補へ移る。
- リスク: 有望候補を保留する機会損失はあるが、現時点でONNX化しても提出候補にならない。

# 2026-06-10 exp238: task130はshape一致のみで保留する

- 背景: exp223でtask130は高costかつcomponent shape full-hit候補だった。
- 新証拠: exp238でcomponent shape matchは `265/265` だが、内容一致は弱い。component exactish `8/265`、max-color/largest crop proxy `0/265`、同shape input crop exact `5/265`。
- 決定: task130を短期のONNX cost probeへ進めない。shape一致だけでは提出候補にならないため、別の高cost mask/shape候補へpivotする。
- リスク: 内容ruleが別に存在する可能性は残るが、現時点の単純proxyでは十分な証拠がない。

# 2026-06-10 exp239-243: task271 ruleは発見、現loweringは不採用

- 背景: exp223でtask271はbaseline `28991`、3x3固定、one_component_shape `266/267` の高cost候補だった。
- 新証拠: exp239-242で、出力は入力内の3x3 full-nonzero block 4候補のうち `color8_count_min` (`sum_colors_min`) のblockであると分かった。Python監査では `267/267`。
- cost証拠: exp243のONNX candidateは validation `267_pass_0_fail` だが cost `70847` でbaseline `28991` を超過。
- 決定: task271をsolved-rule assetに追加するが、現行loweringでは提出しない。4候補だけを安くスコアリングする専用表現ができるまで保留。
- リスク: rule自体はinput-onlyで低leakageだが、Conv+dynamic cropはmemory costが重い。tie-break hidden edgeは将来single-task deltaで確認する。

# 2026-06-10 exp244: task391は色selector未解決で保留する

- 背景: exp223でtask391は3x1固定・stable-binary候補だった。
- 新証拠: exp244でtemplateは3x1全セル同色に固定と分かったが、出力色は単純特徴で説明できずbest `bbox_bl 49/267`。
- 決定: task391をstatic-template loweringへ進めない。色selectorの構造特徴が見えるまで保留。
- リスク: template固定だけで色をtable化するとhidden過学習になる。

# 2026-06-10 exp245-247: task100 ruleは発見、現loweringは不採用

- 背景: exp223でtask100は2x2固定・binary signature 1種類のstable-binary候補だった。
- 新証拠: exp245/246で、出力はbbox_area最大の非zero色による2x2全同色templateと判明し、Python監査 `266/266`。
- cost証拠: exp247のONNX candidateは validation `266_pass_0_fail` だが cost `74548` でbaseline `6536` を大きく超過。
- 決定: task100をsolved-rule assetに追加するが、現行bbox-area loweringでは提出しない。
- リスク: 安いbbox-area selectorが見つかれば再開余地あり。現状のrow/col span計算はmemory costが重すぎる。

# 2026-06-10 exp248/249: task291/task274は短期候補から外す

- 背景: selectorが安そうな低binary候補を継続監査した。
- 新証拠: task291はbest `count_rank2 76/265`、task274は色固定8だがtemplate selector best `70/269`。
- 決定: 両taskを短期のONNX probeへ進めない。次は色固定またはselectorが明確な別候補を探す。
- リスク: task274は固定色なのでtemplate selectorを深掘りすれば解ける可能性はあるが、現時点では証拠が薄い。

# 2026-06-10 exp250: task242は固定templateだが色selector未解決

- 背景: task242はbaseline `20119`、3x3固定、binary template 1種類のため、色selectorだけでstatic loweringできる可能性があった。
- 新証拠: 出力templateは3x3 all-nonzeroで固定だが、色selectorはbest `count_max_low 64/266` と弱い。
- 決定: task242を短期probeへ進めない。次は新規template探索より、解決済みruleの低cost loweringまたは既存artifact surgeryへ戻る。
- リスク: 色selectorはより複雑な構造に依存する可能性があるが、単純特徴では不足。

# 2026-06-10 exp251: small-output単純pruneは効果なし

- 背景: task100/271はrule自体は解けたが、replacement loweringが既存artifactより高costだった。既存artifact側に単純cleanup余地があるか確認した。
- 新証拠: task100/242/253/271の未使用initializerはすべて `0`。costも変化なし。
- 決定: 未使用initializer pruneはこの領域では打ち切り。次は別レーン、またはnode-level redundancyなどより深いsurgeryを体系的に行う。
- リスク: task100の既存artifactは既にruleに近い名前/構造で最適化済みのため、replacementで上回るのは難しい。

# 2026-06-10 exp252/253: task153 full-arc bypass micro deltaを提出する

- 背景: fixed-template新規loweringは不発が続いたため、既存artifact surgeryへ戻した。
- 新証拠: exp252でtask153 `Reshape_node7_to_input0` bypassがfull-arc passし、cost `11212 -> 10947`。
- 決定: exp253としてexp234 current bestにtask153 bypassを載せてKaggle提出する。expected public LB `6006.34`。
- 採点結果: ref `53531156` は public LB `6006.32` でexp234と同点。微小deltaはLB表示上観測されず、best更新なし。
- 決定更新: exp253は採用しない。best baseは引き続きexp234 `6006.32`。
- リスク: 微小deltaは提出枠消費に対して情報量が小さい。以後は同種surgeryでも、複数taskを束ねられる候補か、単独で表示差分が見込めるlocal deltaを優先する。

# 2026-06-10 exp254/255: top30 full-arc bypassを束ねて提出する

- 背景: exp253単独deltaはLB同点で観測されなかったため、同種surgeryを束ねて表示差分が出る大きさにする必要があった。
- 新証拠: exp254でcurrent exp234 top30 cost tasksから13件のfull-arc-pass bypass改善を発見。combined local delta `+0.090241`。
- 決定: exp255として13件をexp234へbundleし、Kaggle ref `53531405` として提出する。expected public LB `6006.4102`。
- 採点結果: public LB `6006.39`。期待値に近い改善を確認したため、current public bestをexp255へ更新する。
- リスク: 13件はすべてfull-arc passだが、graph surgeryの同値性はavailable examples上の証拠。採点結果が出るまでbestはexp234のまま扱う。
- 運用更新: 提出後の採点待ち時間を遊ばせないため、AGENTS.mdに「採点待ち中は次の実験・監査・記録整備を進める」ルールを明文化した。

# 2026-06-10 exp256/257: rank31-80 bypass bundleをexp255へ積む

- 背景: exp255でfull-arc graph-surgery bundleがpublic LBへ転写することを確認した。
- 新証拠: exp256でrank31-80から16件のfull-arc-pass bypass改善を発見。combined local delta `+0.331473`。最大はtask340 `Cast_node0_to_input0`, cost `70342 -> 52342`。
- 決定: exp257として16件をexp255 current bestへstackし、Kaggle ref `53531558` として提出する。expected public LB `6006.7215`。
- 採点結果: public LB `6006.72`。期待値と一致したため、current public bestをexp257へ更新する。
- リスク: exp255/257でレーンの妥当性は上がったが、各taskのhidden edgeは残る。private robustnessは引き続き最終評価まで未確認。

# 2026-06-10 exp258/259: rank81-140 bypass bundleをexp257へ積む

- 背景: exp255/257でfull-arc graph-surgery bundleのLB転写が2回確認された。
- 新証拠: exp258でrank81-140から23件のfull-arc-pass bypass改善を発見。combined local delta `+0.219056`。
- 決定: exp259として23件をexp257 current bestへstackし、Kaggle ref `53531729` として提出する。expected public LB `6006.9391`。
- リスク: rankが下がるほど個別deltaは小さいが、bundle化により表示差分は十分。LB完了までbestはexp257 `6006.72` のまま扱う。

# 2026-06-10 exp262/263: 提出はlocal estimate更新時に限定する

- 背景: graph surgery bundleはpublic LBへよく転写しているが、毎回提出すると提出枠と待ち時間を消費する。
- 新方針: 提出はlocal estimateを更新した時だけ行う。探索だけの結果、no-gain、微小/不確定なpartialは提出せず、bundle replayでfull-arc確認できたものを優先する。
- 新証拠: exp262 partialでrank221-320から17件、local_delta `+0.549899` を回収。exp263で全件再生成・full-arc replay passし、expected public LB `6008.329899` になった。
- 決定: exp263をref `53532720` として提出。採点待ち中は独立した次実験、監査、記録更新を進める。
- 採点結果: public LB `6008.30`。exp261から `+0.52` 改善し、current bestを更新。
- リスク: exp262は途中でtask048付近の評価不安定があったため、今後のrank window sweepはcheckpoint/replay方式を標準にする。

# 2026-06-10 exp264/265: rank321-400 surgeryを提出し、次はdtypeへ戻る

- 背景: exp263でrank221-320までのgraph surgery bundleがLBへ転写したため、残るrank321-400をcheckpoint/replay方式で一巡した。
- 新証拠: exp264で8件、local_delta `+0.597660` を発見。低cost側ではcost差が小さくてもlog score差が大きい。
- 決定: exp265として8件をreplay full-arc確認し、ref `53532884` として提出。expected public LB `6008.897660`。
- リスク: graph surgery window sweepはこれで一巡。以後の主戦場はroadmap Phase B/C(dtype縮小・fused表現)へ戻す。

# 2026-06-10 exp267: farm cost proxyをoutput tensor bytesへ較正する

- 背景: 新しい提出方針では、候補を提出する前にlocal estimateを計算し、submitted best estimateを上回る場合だけ提出する。そのため farm tooling の cost proxy が公式較正とずれていると、Phase B/C候補の選別を誤る。
- 新証拠: 既存 `cost_extractor.py` は中間tensor bytesを足し、最終 `output` を除外していた。exp267で `params + output tensor bytes` に修正し、dtype probeで `uint8=9000`, `float32=36000`, `int64=72000` を確認した。
- 決定: 今後の farm estimate では output dtype を明示的に評価する。exp267自体は候補bundle改善ではないためKaggle提出しない。
- リスク: band判定はまだ古く、低cost op 少数なら `cost_proxy=36000` でも `250-600_plausible` と出る。次の1施策で band threshold を cost proxy 優先に直す。

# 2026-06-10 exp268: farm band判定をcost proxy閾値へ揃える

- 背景: exp267で output bytes proxy は正しくなったが、IR band 判定に低cost op 少数なら `250-600_plausible` とする override が残っていた。
- 新証拠: override削除後、dtype probeは `uint8=9000`, `float32=36000`, `int64=72000` のすべてが `high_cost_probe_only` になり、smokeの `cost_proxy=36000` 候補も high-cost 表示になった。
- 決定: 提出前の farm gate では band 名より `cost_proxy` を優先し、band も閾値に揃える。exp268はtooling較正のみなのでKaggle提出しない。
- リスク: `Where` / `ScatterND` の hard reject はまだ過去guardrailのまま。既知較正の `Where MAC=0` と矛盾する可能性があるため、次は公式cost実測可能なprobeで確認する。

# 2026-06-10 exp269: `Where` をconditional hard rejectから外す

- 背景: 既知較正では `Where` は MAC=0 であり、cost は output tensor bytes と params で見るべきである。従来farmは `Where over full-grid output` を hard reject していた。
- 新証拠: `Where` を `CONDITIONAL_HIGH_RISK_OPS` から外すと、`ONE_NODE_DATA_MOVEMENT` lane の `Where` は `hard_reject=false`, `cost_proxy=9000` になった。一方 `FULL_GRID_COMPOSITION` lane は別理由で reject され続ける。
- 決定: `Where` 自体は hard reject しない。full-grid composition などの構造リスクは別laneのguardで扱う。exp269はtooling較正のみなのでKaggle提出しない。
- リスク: Whereを使う候補の精度/hidden robustness は別問題。実候補では full-arc gate と local estimate 更新を必須にする。

# 2026-06-10 exp270: IR local estimateにparam_countを追加する

- 背景: 既知較正の `recolor_direct(cost 44)` と `recolor_cast(cost 140)` は params 側の差であり、IR path が `param_count=0` 固定だと候補選別に使えない。
- 新証拠: `IRNode.attrs["param_count"]` を加算すると、同じ1-byte outputで `recolor_direct=45`, `recolor_cast=141` と差が local estimate に現れた。
- 決定: IR supplier は小さいparam差を `attrs["param_count"]` で明示する。今後の recolor candidate ranking では direct recolor を cast recolor より優先する。
- リスク: `attrs["param_count"]` はsupplierが正しく付与する必要がある。実候補ではONNX pathの initializer count / official score と突合する。

# 2026-06-10 exp271: accepted candidate rankingをcost-awareにする

- 背景: `recolor_direct` と `recolor_cast` のcost差を計算できても、accepted候補の順序が元順だと同一delta候補で低cost側を優先できない。
- 新証拠: `accepted_candidates()` を `local_delta desc`, `candidate_cost asc` にすると、入力順に関係なく `recolor_direct(cost 45)` が `recolor_cast(cost 141)` より先に来た。
- 決定: bundle候補の基本採用順は local_delta を最優先し、同点では低costを優先する。
- リスク: 同一taskの複数accepted候補を同時に返す点はまだ残る。実bundle化前にはtask単位のbest選択 helper が必要。

# 2026-06-10 exp272: taskごとのbest accepted候補を選べるようにする

- 背景: accepted候補をcost-awareに並べても、同一taskの複数候補をそのままbundleへ渡すと1 task 1 model制約に反する。
- 新証拠: `best_candidates_by_task()` 追加後、accepted 3件のprobeはbest-by-task 2件へ縮約され、`task_recolor` では低costの `recolor_direct(cost 45)` が残った。
- 決定: bundle構築時は `best_candidates_by_task()` を使い、task重複を避ける。`accepted_candidates()` は監査用に全acceptedを返す役割として残す。
- リスク: best選択は local_delta/cost に依存するため、実候補ではlocal_delta計算とofficial scoreの整合確認が必要。

# 2026-06-10 exp273: farm smokeでtask重複を可視化する

- 背景: `best_candidates_by_task()` を追加しても、smoke/result が accepted count だけだと重複taskの有無が見えない。
- 新証拠: `bundle_smoke` に `best_by_task_count` と `duplicate_task_count` を追加し、exp273 smokeで `1/1/0` を確認した。
- 決定: farm result では accepted 全件数だけでなく、task単位のbest件数と重複件数を常に確認する。
- リスク: 可視化だけでは実bundleの重複防止にならない。次はbundle export入口で `best_candidates_by_task()` を使う。

# 2026-06-10 exp274: best-by-task delta合計を分離する

- 背景: `total_local_delta()` がaccepted全件を合計すると、同一taskの複数候補を二重計上し、提出前 local estimate が過大になる。
- 新証拠: duplicate probeでは accepted全件合計 `1.4` に対し、best-by-task合計は `0.9`。重複候補を除くと過大計上を防げる。
- 決定: 提出判断には `best_total_local_delta()` を使う。`total_local_delta()` は全accepted監査用に残す。
- リスク: farm smoke/result にはまだ best total が表示されないため、次に標準可視化へ接続する。

# 2026-06-10 exp275: best_total_local_deltaをfarm resultへ出す

- 背景: 提出判断には dedupe済みの `best_total_local_delta()` を使う方針だが、farm smoke/resultには表示されていなかった。
- 新証拠: exp275 smokeで `best_total_local_delta=0.1` が `bundle_smoke` に出ることを確認した。
- 決定: farm result の提出判断用deltaは `best_total_local_delta` とする。`total_local_delta` は監査用に併記する。
- リスク: `write_notes()` の文面が古いままなので、次に説明を更新して運用ミスを減らす。

# 2026-06-10 exp276: farm notes templateを現在の提出判断方針へ合わせる

- 背景: 実装は output-bytes / Where MAC=0 / best-total-delta 方針へ更新済みだが、`write_notes()` が古い説明のままだと運用ミスを誘発する。
- 新証拠: 更新後の generated notes には `params + output tensor bytes`, `Where` MAC=0, `best_total_local_delta` が明記された。
- 決定: farm notes では accepted全件deltaではなく `best_total_local_delta` をsubmit-gate deltaとして説明する。
- リスク: tooling整備だけではLBは伸びない。次はこのfarm基盤で実候補生成へ戻る。

# 2026-06-10 exp277: 提出可否helperをledgerへ追加する

- 背景: 提出はlocal estimate更新時のみという方針を、人手判断ではなくfarm helperで固定する必要がある。
- 新証拠: `submission_decision(6008.90)` は delta `0.2` で `should_submit=true`、精度gate落ちでdelta `0` なら `false` を返した。
- 決定: 今後の提出可否判断は `best_total_local_delta` ベースの `submission_decision()` を使う。
- リスク: helperをresultへ出さないと運用時に見落とすため、次に farm smoke/result へ接続する。

# 2026-06-10 exp278: farm resultにsubmission_decisionを表示する

- 背景: `submission_decision()` は追加済みだが、実験resultに出なければ提出可否判断の根拠が残らない。
- 新証拠: exp278 smoke resultでは submitted best `6008.90`, best delta `0.1`, candidate `6009.0`, `should_submit=true` が表示された。
- 決定: farm result には提出判断の機械判定を残す。ただし smoke dummy の `should_submit` は実提出対象ではない。
- リスク: smoke dummy と実候補 result を混同する可能性がある。次は smoke-only reason を明示する。

# 2026-06-10 exp279: smoke-only submission decisionを明示する

- 背景: smoke dummy ledgerは合成候補なので、`should_submit=true` でもKaggle提出してはいけない。
- 新証拠: exp279 resultに `submission_decision_scope=smoke_dummy_not_kaggle_candidate` と reason を出せるようになった。
- 決定: smoke result の submission decision は実提出判断ではないことを明示する。実候補では同じ形式を使いつつ scope/reason を実candidate用にする。
- リスク: ここまでtooling整備が続いているため、次は実候補生成へ戻ってscore-producing候補を探す。

# 2026-06-10 exp280: notesにもsubmission decision scopeを出す

- 背景: resultだけでなくnotesにもscope/reasonが出ないと、後から読む時にsmoke dummyの `should_submit=true` を誤読し得る。
- 新証拠: exp280 generated notesで `submission decision scope: smoke_dummy_not_kaggle_candidate` とreasonを確認した。
- 決定: farm notesでも submission decision のscope/reasonを必ず記録する。
- リスク: tooling整備はここで打ち止めに近い。次は実候補生成へ戻る。

# 2026-06-10 exp281: notesにsubmission decision値を出す

- 背景: scope/reasonだけでは、提出判断の数値根拠がnotesから見えない。
- 新証拠: generated notesに submitted best `6008.9`, candidate `6009.0`, `should_submit=True` が表示された。
- 決定: farm notesには submission decision の値とscope/reasonを両方残す。
- リスク: smoke dummy の `True` は実提出対象ではない。scope/reasonとセットで読む。

# 2026-06-10 exp282: IR band判定をcost_proxy基準へ修正する

- 背景: IR `cost_proxy` に params は入っていたが、band判定がmemory bytesだけを見ていたため、params-heavy候補が低costに見えるリスクがあった。
- 新証拠: 修正後、`param_count=5000` / output 1 byte は `cost_proxy=5001` で `high_cost_probe_only` になる。
- 決定: IR band判定は `params + output tensor bytes` の `cost_proxy` に統一する。
- リスク: tooling整備が長く続いた。次は実候補生成でこの判定を使う。

# 2026-06-10 exp283: recolor direct/castをIR primitiveとして区別する

- 背景: `recolor_direct` と `recolor_cast` はcost差が大きく、attrsだけでなくprimitive kindとしても見える方がsupplier/rankingで扱いやすい。
- 新証拠: direct/cast primitiveのprobeで `cost_proxy=45/141` と primitive kind `recolor_direct/recolor_cast` を確認した。
- 決定: recolor候補生成では `PrimitiveKind.RECOLOR_DIRECT` / `RECOLOR_CAST` を使い分ける。
- リスク: primitive追加だけではscoreは伸びない。次は実候補supplierへ戻る。

# 2026-06-10 exp284: farmのpublic-code floor表示に現行best LBを含める

- 背景: `submitted_best_estimate=6008.90` は提出判断に入っているが、`PublicCodeRegistry.floor_status()` の表示が public-code source 由来の古い/空のLBだけになり得る。
- 新証拠: 修正後のsmokeで `current_best_public_lb=6008.9` と `max_observed_kaggle_lb=6008.9` を確認した。
- 決定: public-code floor表示には `EXP_SUMMARY.md` の `Best Public LB` を含め、現行bestと整合させる。
- リスク: 表示整備のみ。`submit_floor_ready=false` なので提出根拠にはならない。

# 2026-06-10 exp285: recolor supplierはdirect-first factoryを使う

- 背景: recolor direct/cast は official cost proxy が `45` と `141` で大きく違うため、supplier入口で区別する必要がある。
- 新証拠: `recolor_direct_program()` と `recolor_cast_program()` のprobeで direct `45`, cast `141` を確認した。
- 決定: recolor候補生成では direct factory を優先し、cast factory は明示fallbackとして扱う。
- リスク: toolingのみでscoreは伸びない。次は実taskに対するcandidate supplierへ進める。

# 2026-06-10 exp286: recolor cost差をfarm smokeに常設する

- 背景: direct-first方針は一度のprobeだけでは後続変更で崩れる可能性がある。
- 新証拠: farm smokeのguardrailに direct `45`, cast `141` が出ることを確認した。
- 決定: recolor supplier接続前に、smokeでdirect/cast cost差を常時監視する。
- リスク: smoke visibilityのみ。score-producing候補生成へ戻る必要がある。

# 2026-06-10 exp287: BundleLedgerのlocal deltaをcost比から計算する

- 背景: `local_delta` 手入力のままだと、提出ポリシーの「params + output bytesからlocal estimateを計算」とズレる。
- 新証拠: `1000 -> 500` probeで `ln(2)=0.693147` が `submission_decision` に反映され、validation fail候補は除外された。
- 決定: ledgerの提出判断は `computed_local_delta=ln(base_cost/candidate_cost)` を使う。
- リスク: CSV互換のため `local_delta` 列は残るが、提出判断では使わない。

# 2026-06-10 exp288: CSV reviewもcomputed_local_deltaを見る

- 背景: `write_csv()` が旧 `local_delta` だけを出すと、提出判定とreview表示がズレる。
- 新証拠: smoke CSVに `computed_local_delta=0.6931471805599453` が出て、`best_total_local_delta` と一致した。
- 決定: candidate reviewでは `computed_local_delta` を提出判断値として見る。
- リスク: `local_delta` 列は互換のため残すが、判断根拠にはしない。

# 2026-06-10 exp289: selected_manifestをBundleLedgerへ直接取り込む

- 背景: real candidate を提出判定に流すには、既存 sweep が出す `selected_manifest.csv` を ledger 化する入口が必要。
- 新証拠: exp264 manifest 8件をacceptedし、cost-derived delta `+0.5976596441240898` を再現した。
- 決定: 今後の selected manifest review は `BundleLedger.from_selected_manifest()` で ledger 化し、提出判定へ接続する。
- リスク: 既提出manifestを新規改善として扱わない。履歴deltaは現行bestに含まれるかを必ず確認する。

# 2026-06-10 exp290: 複数selected_manifestをまとめてledger化する

- 背景: real bundle reviewでは複数sweepのmanifestをまとめ、同一task重複をbest-by-taskで処理する必要がある。
- 新証拠: exp260+exp264 manifestをまとめて35件acceptedし、`best_total_local_delta=1.4377463495672809` を計算できた。
- 決定: 複数manifest reviewは `BundleLedger.from_selected_manifests()` を使う。
- リスク: 履歴manifestの `should_submit=true` は新規提出根拠ではない。current-best lineage確認を必須にする。

# 2026-06-10 exp291: current-best lineage sourceを除外して提出判定する

- 背景: 履歴manifestをまとめると既に提出済みのdeltaで `should_submit=true` になり得る。
- 新証拠: exp260+exp264 sourceを除外すると `candidate_count=0`, `should_submit=false` になった。
- 決定: real manifest reviewでは現行best lineage の `source_exp` を `exclude_sources` に入れてから提出判定する。
- リスク: 除外sourceの指定漏れ。current best lineageをEXP_SUMMARY/LB_Trackingと照合する。

# 2026-06-10 exp292: submission decisionに候補件数を含める

- 背景: `should_submit=false` の理由がfresh候補ゼロなのか、deltaゼロなのかをdecisionから直接読める必要がある。
- 新証拠: lineage除外後のdecisionで `accepted_count=0`, `best_by_task_count=0`, `should_submit=false` を確認した。
- 決定: 提出判定では `accepted_count` / `best_by_task_count` / `best_total_local_delta` をセットで確認する。
- リスク: fresh候補ゼロが確認できたため、次はtoolingではなくscore-producing候補生成へ戻る。

# 2026-06-10 exp293: fresh-only提出判定を1関数にまとめる

- 背景: real candidate manifest生成後に、lineage除外と提出判定を別々に呼ぶと運用ミスが起き得る。
- 新証拠: `fresh_submission_decision()` で履歴manifest6件を入力しても、lineage除外後は `accepted_count=0`, `should_submit=false` になった。
- 決定: 新規manifest reviewでは `BundleLedger.fresh_submission_decision()` を使う。
- リスク: helper整備はここで打ち止め。次はscore-producing候補生成に戻る。

# 2026-06-10 exp294: full_arc_pass表記もaccuracy gate passにする

- 背景: 既存manifestは `full_arc_pass`、ledger内部は `*_pass_0_fail` を主に使っており、direct supplierで表記揺れが起きる。
- 新証拠: synthetic probeで `full_arc_pass` と `266_pass_0_fail` はaccepted、`24_pass_1_fail` はrejectedになった。
- 決定: `BundleCandidate.accepted` は `full_arc_pass` と `*_pass_0_fail` をaccuracy passとして扱う。
- リスク: `full_arc_pass` は本当にfull-arc validation済みの候補にのみ使う。

# 2026-06-10 exp295: exp262の残りwindowはtask048で止まる

- 背景: fresh manifest が0件だったため、score-producing候補生成へ戻る必要があった。exp262 rank221-320 はpartialで止まっており未消化余地がある。
- 新証拠: 既存script再実行でも `after_task273` の17 selectedまでで、その後 task048 付近のONNXRuntime reshape/conv errorにより非ゼロ終了した。
- 決定: exp262 partialの17件は現行best lineageに含まれるため再提出しない。次はtask048 bad candidatesをskip/quarantineして残りwindowを完走する。
- リスク: script本体修正は今回のallowed-file制約外。次ループで制約を満たす実装経路を選ぶ。

# 2026-06-10 exp296: task-level quarantineをfresh submit reviewに入れる

- 背景: task048のような既知runtime-stopperを提出reviewからも明示除外できる必要がある。
- 新証拠: synthetic manifestで `exclude_tasks={"48"}` が効き、全task除外時は `should_submit=false` になった。
- 決定: fresh manifest reviewでは必要に応じて `exclude_tasks` を使い、既知危険taskを候補集合から落とす。
- リスク: これはreview側の安全弁であり、exp262生成scriptがtask048で止まる問題自体は未解決。

# 2026-06-11 exp329: task185 full compact candidateを次の主軸にする

- 背景: exp323/325 の fresh public-zero probe は連続 all-alive で、`franksunp_blended_best` 高 point 順の収率が落ちた。task037 は exp327 で correctness は通ったが dense lowering cost wall に当たった。
- 新証拠: exp328 で task185 dynamic bg detector が cost `143`、exp329 で detector-connected axis selector が `267_pass_0_fail` かつ cost `41519` vs baseline `59584` で成立した。
- 決定: 強い新 public-zero suspect source が出るまでは、次の score-direct 主軸を task185 full compact candidate に置く。
- リスク: exp329 は selector subgraph probe であり、selected lattice extraction と homogeneous 2x2 core を接続した full candidate は未検証。exp204 の巨大 template 再発を避ける。

# 2026-06-11 exp330: task185は次1本だけcost shaveしてからpivot判定

- 背景: exp330 で task185 full candidate が `267_pass_0_fail` まで通ったが、best cost は `59784` で baseline `59584` より `200` 高かった。
- 新証拠: compact start/spacing dynamic index は exp204 の巨大 template 問題を解消した。final `uint8` output variant は cost `95784` で悪化し、output dtype Cast は短期解ではない。
- 決定: 次の1実験だけ task185 の narrow cost shave を試す。default-bg + Pad、Tile index shape、selector intermediate の削減で `200` 以上削れなければ task251/task037 へ pivot する。
- リスク: task185 に長居しすぎると score-producing でない微調整が続く。次実験の中止条件を `cost < 59584` 未達に固定する。

# 2026-06-21 exp338(P0-1): 採点校正はORT実行ベース採用＋7件freeze、絶対校正はP1-2へ

- 背景: union zip のローカル公式再採点 7023.58 が Kaggle 7117.01 を93点下回る。原因は7 artifact の
  負pads(Conv/ConvTranspose)が local onnx==1.22.0 の check_model(full_check=True)/
  infer_shapes(strict_mode=True) で ShapeInferenceError → sentinel 化。ORT 実行と strict=False は成功。
- 新証拠: 7件は安価(cost 100〜1993, pts 17.4〜20.4 ≈計125点)で高コストではない。3案のうち
  負pads正規化は artifact cost を変え校正破壊、strict迂回は memory 過小計上＋task149 未復旧。
  ORT実行(trace)ベース採点のみが全件復旧かつ堅牢。
- 決定: (1) neurogolf_calib.score_model_calibrated(公式優先＋負pads時のみtraceフォールバック)採用、
  公式 neurogolf_utils.py 無改変、393件 byte 一致。(2) baseline=7147.94/sentinel0 を採用(旧版.bak退避)。
  (3) climb sweep で7件 freeze(trace法 baseline と公式法 candidate 非整合＋Kaggle真コスト未校正)。
  (4) auto-submit は P1-2 まで OFF。
- 重大(前提崩れ): baseline 7147.94 は Kaggle 7117.01 を +31 上回り §8 [7116,7118] 到達不能。
  残差は onnx 版差の広域僅差で絶対校正は提出観測でしか確定不能 → 完了条件再定義を提案、要協議。
- リスク: leakage なし(採点器のみ)。overfit は7件 freeze で遮断。

# 2026-06-21 exp339(P1-1/P-Ops): phase1 builderは union未満→Phase2が本丸、自律統合はall-green

- 背景: P0-1完了後、16 builder全投入で1 sweep。
- 新証拠: 393 improvable で **0改善**。builderは full-arc pass 候補を生成するが全て union より高コスト
  (task187 237631>78309 等)。→ **公開7117 union が phase1 builder を全タスクで支配**。
- 決定:
  1. Phase1(既存builder再合成)は点にならないと確定。実利得は **Phase2 桁落とし** に集約。
     次の最速候補は **P2-3 dtype最小化を層Cに追加し union artifact 自体を桁削り**。
  2. P-Ops: climb を `run_climb()` 化し `autonomous_runner --mode climb` / `monitor -Mode climb` へ統合。
     crash-resilience=monitor再起動+ledger永続resume、sweep内gc、heartbeat、lb_tracking。
  3. auto-submit は ON(ユーザー選択)。但し単調ゲートにより無改善時は提出されない(検証済)。
- リスク: 無し(提出は真の局所改善時のみ。floor 7117 を割らない単調設計を維持)。overfit は full-arc(-1)で排除。

---

## 2026-06-21 【撤退判断確定】neurogolf-2026 自前研究を終了

- **決定**: 自前のスコア改善研究を終了する。**floor 7117.01（提出済・順位確定・安全）を最終成果**とする。
- **根拠（実測）**: 安全な一般手段を全て試し全0win（exp338 P0校正 / exp339 P1 16builder / exp340 P2 dtype・const-fold・prune / exp341 P2 emitter+GridSample全393scan / exp342 P3 enumerative solver 2.8%・LLM proposer）。**公開7117 union は per-task で十分 golf 済**で、安全な汎用変換では超えられないと確定。
- **残る前進路と却下理由**: bespoke per-task 再lowering（豊かなDSL×安価static lowering）のみ。これは**ARC Prize 級の大型研究**で、過去 exp317/327/332 も union 未達・工数数週間・成功確率低・deadline 7/16 迫る。**限界ROIが低く、他コンペ（rogii / signate-nir / PTGC）の機会費用に劣る**と判断。
- **今後の運用**: 自前研究・自律climb常駐は停止。**唯一の高ROI行動として「公開Notebook/データで 7117 超が出たら即コピー採用」する薄い監視のみ**を残す（コピー支配の field であり、7117 到達も本質これ）。
- **資産**: 5層フレーム（calibrated scorer / monotone gate / emitter拡張 / solver / LLM proposer / 規則化計測）は安全に整備済。再開時は `docs/ARCHITECTURE_TO_RANK1.md` と本ログを参照。
- **不変条件は維持**: floor 非回帰・full-arc gate・union/baseline 無変更・提出ゼロ（自前研究中）。
