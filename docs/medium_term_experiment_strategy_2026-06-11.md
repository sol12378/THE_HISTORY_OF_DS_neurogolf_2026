# NeuroGolf 中長期実験戦略分析 2026-06-11

## 目的

Public LB 7000+ を中長期的に狙うため、これまでの実験ポートフォリオを投資判断の観点で整理する。

単なる時系列まとめではなく、今後どの実験系統へ時間を投資し、どの系統を縮小・停止するかを決めるための分析である。

## 1. 現状サマリー

### 現在地

| 指標 | 値 | 根拠 |
|---|---:|---|
| Best Public LB | `6008.96` | exp297 / Kaggle ref `53535771` |
| Best Submit-Safe | `5930.55` | exp127 / exp135 confirmation / ref `53480507` |
| Best Strict Bundle | local `6282.23` | exp005 |
| Best Local Upper Bound | local `6480.30` | exp041。ただし LB collapse 済み |

現 best の exp297 は、public-zero repair、task300 rule lowering、full-arc graph surgery / bypass を積み上げたもの。Public LB には効いているが、public blend 由来 raw を含むため private robustness risk は submit-safe seed より高い。

### 実際に LB に効いた系統

| 系統 | 主な成果 |
|---|---|
| Public-zero repair | 約 `+75` LB。task018/023/025/133/158/285 など |
| full-arc graph surgery / bypass | exp255-297 で約 `+2.68` LB |
| task-specific rule lowering | task020 `+0.21`、task300 `+0.39`、task025 repair `+11.90` |
| calibration / bisection probe | 直接 gain ではないが zero 特定に最大貢献 |

### 詰まり始めている系統

- `franksunp_blended_best` 高 point 順の public-zero probe は exp323/325 で連続 all-alive となり、直近収率が低下している。
- dense ONNX lowering は正しい rule でも cost wall に当たりやすい。exp327 task037 は full-arc pass したが cost `6633317` で不採用。
- broad archetype scan / generic rule scan は improved 0 が多く、投資効率が低い。
- farm/tooling 先行整備は exp267-313 で長く続いたが score-producing ではなかった。今後は tooling 単体を主目的にしない。

### 未解決だが価値が高い候補領域

1. GridSample queue: task251 / task037 / task185 / task048
2. task-specific solved-rule lowering: task185, task085, task365, task366, task346, task271, task100
3. L4 shape/crop family 圧縮
4. L2 sparse fill family 圧縮
5. 残 public-zero の再探索。ただし候補選定を改善する必要がある。

## 2. 実験系統ごとの評価

### Public-zero repair

| 項目 | 内容 |
|---|---|
| 代表 exp | exp139-192, exp315-325 |
| 成果 | 最大実績レーン。bisection probe の expected drop と LB が小数点2桁でほぼ一致し、task 単位 zero を特定できた |
| 失敗パターン | exp323/325 は all-alive。exp321 は task101/task243 repair が no-gain |
| 再投資判断 | 条件付き継続。高 point 順ではなく、source provenance / subset-sum uniqueness / family diversity を使う |
| 次に検証 | 未 probe task から一意性の高い 12-16 task probe group を作る |
| leakage risk | low to medium。fail-stub probe 自体は診断だが、public feedback 依存 |
| overfitting risk | medium。public-zero repair は public LB に近いが、local full-arc correctness を必須にする |

### GridSample / ONNX graph surgery

| 項目 | 内容 |
|---|---|
| 代表 exp | exp300-314, exp324-328 |
| 成果 | GridSample は実 ONNX で cost `1800` と scoreable。task251/037/185/048 の queue が整理された |
| 失敗パターン | task251 は純 sampling では color1 を生成できない。task037 dense shift は cost 爆発 |
| 再投資判断 | 強く再投資。低コスト表現の本命 |
| 次に検証 | task185 dynamic-bg selector、task251 mask + cheap recolor、task037 sparse diagonal table |
| leakage risk | low。input-only rule / geometry が中心 |
| overfitting risk | low to medium。task-specific だが local examples 全体で検証可能 |

### task-specific rule lowering

| 項目 | 内容 |
|---|---|
| 代表 exp | exp066, exp166, exp234, exp327, exp328 |
| 成果 | task020/task025/task300 は LB 換金済み。task185 の dynamic bg detector は cost `143` で成立 |
| 失敗パターン | correct rule でも lowering cost が baseline を超える |
| 再投資判断 | 継続。汎用探索ではなく 1 task 深掘り型で進める |
| 次に検証 | task185 detector-connected selector/core、task085, task365/366 の小出力表現 |
| leakage risk | low to medium |
| overfitting risk | medium。task-specific rule は妥当だが、public-only 最適化にならないよう local full-arc を必須にする |

### program synthesis

| 項目 | 内容 |
|---|---|
| 代表 exp | exp017-062, exp_b 系 |
| 成果 | DSL、rule backlog、task family taxonomy、task020 分解などの知見を蓄積 |
| 失敗パターン | 汎用探索は full hit が少なく、直接 LB に効きにくい |
| 再投資判断 | 自動探索の主戦場化はしない。task-specific 補助に降格 |
| 次に検証 | L2 sparse fill の上位 5 task に限定した半自動 rule decomposition |
| leakage risk | low to medium |
| overfitting risk | medium。train-only rule compression は要注意 |

### cost optimization

| 項目 | 内容 |
|---|---|
| 代表 exp | exp032-041, exp252-297 |
| 成果 | full-arc bypass は安定して Public LB に転写。107 task / 約 `+2.68` |
| 失敗パターン | 収穫逓減。exp297 fresh remainder は `+0.055` 程度 |
| 再投資判断 | filler 限定。主戦場にはしない |
| 次に検証 | pending 中に 2周目 bypass を小さく回す程度 |
| leakage risk | low |
| overfitting risk | low to medium。full-arc gate は効くが、delta が小さい |

### validation / leakage audit

| 項目 | 内容 |
|---|---|
| 代表 exp | exp056, exp135-137, exp298, exp322 |
| 成果 | local/LB gap と public-zero の構造を解釈。submit 判断の精度を上げた |
| 失敗パターン | full-arc pass でも public-zero raw があるため、validation だけで source を信用できない |
| 再投資判断 | 継続必須。ただし audit-only を長く続けない |
| 次に検証 | expected LB と実 LB の乖離を task 単位で即診断する |
| leakage risk | low |
| overfitting risk | low to medium |

### submission blending / routing

| 項目 | 内容 |
|---|---|
| 代表 exp | exp002, exp008, exp130, exp_b035 |
| 成果 | exp_b035 raw は複数 public-zero repair に成功 |
| 失敗パターン | 丸ごと public CODE / public blend 採用は collapse |
| 再投資判断 | repair source としてのみ使う |
| 次に検証 | b035 依存 task の自前 rule 置換 |
| leakage risk | high |
| overfitting risk | high |

## 3. タスク単位の分析

### すでに強く改善済みの task

- Public-zero repair 済み: task018, task023, task025, task133, task158, task285
- rule 換金済み: task020, task300
- graph surgery / bypass 済み: exp254-297 の rank sweep 対象 107 task

### まだ Public LB 上の未修復候補がありそうな task

- task187 は以前は強い候補だったが、exp317 で current-best lineage では alive らしいと判明した。
- task101/task243 は exp319 fail-stub では zero 風だったが、exp321 repair が no-gain。素直な repair target としては扱いにくい。
- 今後の候補は、単純な高 point 順ではなく、未 probe task の source provenance、subset-sum uniqueness、family diversity で選ぶべき。

### validation では promising だが LB に出ていない task

| task | 状態 |
|---:|---|
| 251 | Python rule full pass。GridSample 単体は不可。mask + recolor が必要 |
| 037 | Python rule full pass。dense lowering は correct だが高コスト |
| 185 | dynamic bg detector が cost `143` で成立。selector/core 接続が次 |
| 048 | bridge connectivity rule 資産あり。runtime-stopper と lowering 課題 |
| 085 | horizontal bar alternate erase rule 資産あり |
| 365/366 | object/crop/copy 系 rule 資産あり |
| 346/271/100 | component / selector 系の promising audit あり |

### ONNX コストが障害になっている task

- task037: full-arc pass したが dense shift-stack cost `6633317`
- task187: correctness-first repair cost `356431`
- task251: pure GridSample は安いが、closed component mask が課題
- task031/task398/task203/task221: GatherND / ScatterND / MatMul / Tile 系の cost wall

### 実装可能性はあるが工数が重い task

- task251: closed component mask + color1 generation
- task037: compact diagonal representation
- task185: dynamic axis selector + output core
- L4 shape/crop family: GridSample / index-map 表現が確立すれば大きい
- L2 sparse fill family: task020 型の rule decomposition が必要

## 4. 失敗実験からの学び

### 仮説が間違っていた

- subset-sum だけで zero 集合を決められる、という見方は弱かった。probe が必要。
- public artifact を local full pass だけで信用する仮説は exp041/130 で崩れた。

### 実装表現が高コストすぎた

- full-grid intermediate、巨大 Slice/Pad stack、大 index tensor、dynamic MatMul / GatherND は繰り返し失敗。
- 正しい rule と安い lowering は別問題。task037/187/251 が代表例。

### validation と public distribution がずれた

- exp143, exp186, exp321 では full-local-valid raw が Public LB に効かなかった。
- full-arc pass は必要条件であって十分条件ではない。

### probe の情報価値はあった

- exp139-190 と exp315-325 は、best LB を上げない提出でも alive/zero の探索空間を大きく削った。
- expected LB と observed LB の一致/不一致が task 単位診断として機能している。

### 今後再利用できる部品

- bisection probe builder
- full-arc gate
- bundle ledger
- task quarantine
- dynamic bg detector
- GridSample cost probe
- task-specific rule audit workflow

## 5. 中長期戦略

### Phase A: 1日以内にやるべき高期待値実験

目的は、即座にスコア直結しうる候補の詰まりを解くこと。

| 優先 | 実験案 | 成功条件 | 中止条件 | 想定 LB gain |
|---:|---|---|---|---:|
| 1 | exp329_task185_dynamic_bg_axis_selector_probe | bg/row/col selector が全例一致し、cost が baseline 未満 | selector 不一致が構造的なら診断で止める | 後続 +1〜4 |
| 2 | exp330_task185_compact_core_candidate | full-arc pass、cost < task185 baseline | cost > baseline | +1.6〜4.6 |
| 3 | exp331_public_zero_uniqueness_probe_d | all-alive 以外、missing drop が解読可能 | 2連続 all-alive なら再停止 | 後続 +10〜30 |

主なリスクは、task185 が selector で解けても output core が高コストになること、public-zero probe がさらに all-alive になること。

### Phase B: 2-4日で積み上げるべき実験群

目的は、GridSample queue と solved-rule queue を提出可能な単 task candidate にすること。

優先順位:

1. task185 を提出可能 candidate にする。
2. task251 の mask + cheap recolor を試す。
3. task037 は dense shift を捨て、sparse changed-cell / diagonal table に移る。
4. task048 は runtime-stopper と GridSample 候補の二方向で audit する。

成功条件:

- 1-2 task で full-arc pass + cost improvement
- expected LB と実 LB の一致確認

中止条件:

- 3 task 連続で correct だが cost wall
- GridSample queue 4 target 中 3 target が構造的に不成立

想定 LB gain は `+5〜15`。

### Phase C: 1週間以上の中期投資候補

目的は LB7000+ に必要な family 圧縮。public-zero と単 task 改善だけでは 6200-6400 付近で止まる可能性が高い。

重点:

1. L4 shape/crop family に GridSample / small index map を適用する。
2. L2 sparse fill family に task020/task251 型の rule decomposition を適用する。
3. b035 依存 repair task を自前 rule へ置換して private risk を下げる。
4. farm は候補生成に使い、tooling 単体実験は凍結する。

成功条件:

- 10 task 以上で cost <= 2000 級
- または 5 task 以上で実 LB 転写

中止条件:

- 15 task 試して 2 task 未満の改善
- rule searcher が task あたり 5 実験以上かかり、量産性が出ない

想定 LB gain は `+100〜500`。LB7000 の本丸。

## 6. 意思決定ルール

### Kaggle submit する条件

- diagnostic bisection probe は、対象集合・expected all-alive LB・subset-sum 解読計画が保存済みなら submit 可。
- repair は full-arc pass かつ expected LB gain が明確なら submit。
- cost improvement bundle は local delta > `0.05` かつ full-arc replay pass なら submit。
- public/raw source は単 task または小 bundle で較正し、丸ごと採用しない。

### local probe で止める条件

- detector / selector / cost probe のような中間 subgraph。
- full-arc pass していない。
- cost が baseline を超え、public-zero repair でもない。
- correctness はあるが dense lowering で cost wall が見えている。

### 系統ごと捨てる条件

- broad scan で 5実験連続 full-hit なし。
- GridSample queue 4 target 中 3 target が構造的に不成立。
- public-zero probe が uniqueness selection 後も 3回連続 all-alive。
- tooling-only が発生しそうなときは、score-direct requirement がない限り止める。

### pending submission 中に進めるべきこと

- pending score に依存しない次 probe group の設計
- repair source audit
- task-specific Python rule / ONNX subgraph probe
- notes / LB_Tracking / Decision_Log 更新
- full-arc validation scripts の実行

### public LB に過剰適合しないための制約

- 最終提出は safe bundle と challenger を分ける。
- public blend raw は防御的に自前 rule へ置換する。
- full-arc pass は必要条件であって十分条件ではないと扱う。
- public-zero repair は local correctness を必ず伴わせる。

## 7. 次の 5 実験

| exp案 | 仮説 | 実装内容 | 期待 gain | 所要時間 | 成功条件 | submit | 記録観点 |
|---|---|---|---:|---:|---|---|---|
| exp329_task185_dynamic_bg_axis_selector_probe | dynamic bg を除外すれば axis selector が全例一致する | exp328 detector + dilated row/col scorer | 中間。後続 +1〜4 | 20-40分 | bg/row/col selector 全例一致、cost < baseline | no | selector mismatch、cost、leakage/overfitting |
| exp330_task185_compact_core_candidate | selector が合えば task185 全体を baseline 未満にできる | selected lattice extraction + compact output core | +1.6〜4.6 | 1-2h | full-arc pass、cost < `59584` 目安 | yes if pass | expected LB、official cost、core failure |
| exp331_task251_mask_recolor_min_probe | GridSample 単体では無理だが mask + color1 合成なら勝てる | closed component mask 最小化、cheap recolor | +2〜4 | 1-2h | full-arc pass、cost < baseline `100580` | yes if pass | mask cost、color generation、risk |
| exp332_task037_sparse_diagonal_table_probe | dense shift ではなく changed-cell sparse 表現なら cost wall を避けられる | endpoint pair table / sparse ScatterND 最小候補 | +2〜4 | 1-2h | full-arc pass、cost < baseline `63726` | yes if pass | changed-cell count、table size、cost |
| exp333_public_zero_uniqueness_probe_d | 高point順でなく uniqueness/risk 選定なら zero 再発見率が戻る | 未probe task 12-16件 fail-stub、subset-sum 一意性つき | 後続 +10〜30 | 30-60分 | expected/observed から解読可能 | yes | expected all-alive LB、subset-sum candidates |

## 結論

明日からの主戦場を 1 つだけ選ぶなら、task185 を軸にした GridSample / compact lowering lane。

捨てるべき実験系統は、broad archetype scan と tooling-only farm 整備。

一見地味だが中長期で効く整備は、各候補の `expected LB`、`full-arc status`、`official cost`、`source risk` を同じ ledger で比較し、submit 判断を機械的にすること。

