# NeuroGolf 実験ポートフォリオ分析と中長期計画

> 作成日: 2026-06-11  
> 目的: Public LB 7000+ を中長期的に狙うため、これまでの実験ポートフォリオを投資判断の観点で整理する。  
> 参照: `AGENTS.md`, `EXP_SUMMARY.md`, `docs/experiment_plan_2026-06-10.md`, `docs/experiment_portfolio_analysis_2026-06-11.md`, `neurogolf_kaggle_obsidian/`, `experiments/exp*/result.json`, `experiments/exp*/notes.md`

---

## 1. 現状サマリー

### 1.1 現在の best

| 区分 | 実験 | 値 | 解釈 |
|---|---|---:|---|
| Best Public LB | `exp331_task185_gather_axis_cost_shave` | `6009.15` | 現在の採用基準。exp297 に task185 compact Gather lowering を足した bundle。以後はこの bundle を壊さず差分で追加 gain を狙う。 |
| Best Submit-Safe Delta | `exp127_focused_surgery_seventh_pass_limited` / `exp135` | Public LB `5930.55` | public artifact 依存を抑えた比較的安全な基準線。 |
| Best Strict Local | `exp005_top_cost_rewrite_strict` | local `6282.23` | strict full validation pass。ただし LB は `5929.89` で gap が大きい。 |
| Best Local Upper Bound | `exp041_task145_deeper_mul_chain` | local `6480.30` / LB `3417.71` | lookup/public artifact overfit の代表的な失敗例。 |

主要 submission の流れは、`exp005/exp127` の strict seed から、public-zero repair と full-arc graph surgery を積み上げて `exp297` の `6008.96` まで到達し、さらに `exp331` の task185 compact Gather lowering で `6009.15` へ進んだ、という構図である。06-10 の伸びは大きかったが、06-11 の `exp323` / `exp325` では fresh wide public-zero probe が連続 all-alive となり、同じ選定方法の限界が見え始めている。

### 1.2 実際に LB に効いた系統

| 系統 | LB 効果 | 代表実験 | 評価 |
|---|---:|---|---|
| Public-zero repair | 最大 | `exp144`, `exp152`, `exp166`, `exp172`, `exp178` | 0 点 task を点に戻すため、cost が高くても成立する。これまで最も収益性が高い。 |
| Full-arc graph surgery / bypass | 小さいが堅い | `exp255`, `exp257`, `exp259`, `exp261`, `exp263`, `exp265`, `exp297` | expected LB と実 LB がほぼ一致。堅いが逓減済み。 |
| task-specific rule lowering | 中程度 | `exp066`, `exp166`, `exp234` | rule が正しく、ONNX lowering が cost 競争力を持つ場合は LB へ転写する。 |

### 1.3 詰まり始めている系統

- `franksunp_blended_best` 高 point 順 public-zero probe: `exp323` / `exp325` が連続 all-alive。高期待値候補はかなり掃かれた可能性が高い。
- Dense full-grid lowering: `task037` の `exp327` は full-arc valid だが cost `6633317`。正しい rule でも表現が重いと使えない。
- 横断的 broad scan / archetype scan: 多数の no gain 実験があり、task-specific 診断なしに広げる価値は低い。
- Tooling-only farm 整備: `exp267` 以降で十分整備済み。これ以上の先回り実装は score へ直結しにくい。

### 1.4 未解決だが価値が高い候補領域

| 候補 | 期待値 | 現状 |
|---|---:|---|
| 未特定 public-zero task | 中から大 | 初期は高収益だったが、直近 probe の収率は低下。強い suspect source または subset-sum 一意性の高い group 設計が必要。 |
| `task185` compact lowering | 済み | `exp331` で `267_pass_0_fail`、cost `59584 -> 48889`、Public LB `6009.15`。dynamic index 縮小の成功例として他 task に再利用する。 |
| GridSample / compact data movement | 中 | 実 ONNX GridSample は `exp324` で cost `1800` と判明。ただし task251 は新色生成があり pure sampling では不可。 |
| `task037` compact diagonal representation | 中 | rule は full pass。dense lowering は不可。per-diagonal / sparse changed-cell 表現なら再投資余地。 |
| L4 shape/crop / L2 sparse fill family | 大だが長期 | LB7000+ の本丸。ただし task-specific lowering の成功例を先に増やす必要がある。 |

---

## 2. 実験系統ごとの評価

### 2.1 Public-zero repair

| 観点 | 評価 |
|---|---|
| 代表 experiment id | `exp139-190`, `exp144`, `exp152`, `exp166`, `exp172`, `exp178`, `exp315-325` |
| 成果 | task018/023/025/133/158/285 などを repair し、Public LB を `5930.55` から `6005.93` 以上へ押し上げた。fail-stub probe の expected drop は非常に再現性が高い。 |
| 失敗パターン | `exp321` の task101/task243 は fail-stub で zero らしく見えたが、`franksunp_blended_best` repair では LB gain なし。`exp323` / `exp325` は全て alive。 |
| 再投資判断 | 条件付きで継続。高 point 順を漫然と続けるのではなく、source risk・subset-sum 一意性・未 probe family を見て group を設計する。 |
| 次に検証すべきこと | 新しい suspect source の発見、probe group の subset-sum 一意性、public-zero らしさの説明変数。 |
| leakage / overfitting risk | 中。probe 自体は診断なので leakage 低めだが、public artifact raw を repair として使う場合は private robustness risk が高い。 |

### 2.2 GridSample / ONNX graph surgery

| 観点 | 評価 |
|---|---|
| 代表 experiment id | Graph surgery: `exp032-041`, `exp109-127`, `exp252-265`, `exp297`; GridSample: `exp300-314`, `exp324` |
| 成果 | graph surgery は LB 転写が堅く、`exp255-297` で小幅に積み上げた。GridSample は実 ONNX cost `1800` が確認され、低 cost primitive として有望。 |
| 失敗パターン | surgery は fresh delta が `+0.055` まで逓減。GridSample はまだ実 task の正解 candidate を作れていない。task251 は新色生成が障害。 |
| 再投資判断 | surgery は採点待ち filler。GridSample は task-specific に絞って再投資。 |
| 次に検証すべきこと | `task037` の sparse diagonal、`task251` の mask + color1 合成、task185 で効いた `Gather(axis=2/3)` 型 index 縮小の横展開。 |
| leakage / overfitting risk | graph surgery は full-arc gate 付きなら低から中。GridSample は rule-driven なら低いが、public artifact 由来の座標 table 化は中から高。 |

### 2.3 task-specific rule lowering

| 観点 | 評価 |
|---|---|
| 代表 experiment id | `exp066`, `exp166`, `exp234`, `exp316-330` |
| 成果 | task020/task025/task300/task185 は実 LB に転写。task037/task251 は rule は解けている。 |
| 失敗パターン | 正しい rule を full-grid intermediate で実装すると cost wall に当たる。`exp327` が典型。 |
| 再投資判断 | 高い。Public LB 7000+ にはこの系統の量産が必要。ただし 1 task につき cost-aware lowering の見込みを早く判定する。 |
| 次に検証すべきこと | `task037` の compact diagonal、`task251` の component mask 最小化、task185 の compact Gather pattern を他 task に使えるか。 |
| leakage / overfitting risk | 低から中。入力から説明可能な rule なら低い。train/test/arc-gen の特殊性に寄せた selector は中。 |

### 2.4 Program synthesis

| 観点 | 評価 |
|---|---|
| 代表 experiment id | `exp017`, `exp018`, `exp022`, `exp026`, `exp042`, `exp045`, `exp054-062` |
| 成果 | DSL backlog、family taxonomy、rule decomposition の考え方を確立。task020 では rule 分解の成功パターンを作った。 |
| 失敗パターン | 汎用 synthesis は fit しても ONNX cost が高い、または full-arc generalization が弱い。 |
| 再投資判断 | 中期投資。短期 LB gain より、L2/L4 family の量産基盤として使う。 |
| 次に検証すべきこと | 1 task あたり 1-2 実験で rule 仮説から lowering まで進めるワーカー運用。 |
| leakage / overfitting risk | 中。lookup 的 table 化に寄ると高い。説明可能な feature/rule へ圧縮できるかが分岐点。 |

### 2.5 Cost optimization

| 観点 | 評価 |
|---|---|
| 代表 experiment id | `exp005`, `exp009`, `exp023`, `exp027`, `exp158-161`, `exp193-194`, `exp266-267`, `exp330` |
| 成果 | unused initializer prune、safe scalarization、dtype cost の理解、full-grid / int64 index の危険性を guardrail 化。 |
| 失敗パターン | 汎用 optimizer はほぼ no gain。final output dtype Cast は `exp330` で cost 悪化。 |
| 再投資判断 | 個別 candidate の blocker 解消としてのみ継続。汎用 post-pass への大投資は避ける。 |
| 次に検証すべきこと | `GatherElements` の大 index tensor を `Gather axis` に置換できるか。Pad/default-bg の cost を削れるか。 |
| leakage / overfitting risk | 低。ただし cost だけに寄せて validation を弱めると functional failure risk が上がる。 |

### 2.6 Validation / leakage audit

| 観点 | 評価 |
|---|---|
| 代表 experiment id | `exp006`, `exp056`, `exp136-140`, `exp315-322` |
| 成果 | local/LB gap の構造、public artifact collapse、probe の有効性を明確化。 |
| 失敗パターン | subset-sum 算術だけで zero task を断定すると外れる。full-arc pass でも public/private mismatch は残る。 |
| 再投資判断 | 必須。新 source raw を採用する前には必ず audit / probe を通す。 |
| 次に検証すべきこと | b035 由来 repair task の stress validation、自前 rule 置換の優先順位。 |
| leakage / overfitting risk | audit 自体は低いが、Public LB feedback を使いすぎると public split への過適合が進む。 |

### 2.7 Submission blending / routing

| 観点 | 評価 |
|---|---|
| 代表 experiment id | `exp002`, `exp008`, `exp130`, `exp289-297` |
| 成果 | bundle ledger / lineage exclusion / fresh submission decision が整備され、既存 best を壊さず差分採用できる。 |
| 失敗パターン | public blend の丸ごと採用は LB collapse。source 単位で信用せず task 単位に分解する必要がある。 |
| 再投資判断 | 運用基盤として維持。新規 tooling は実候補が必要とした時だけ。 |
| 次に検証すべきこと | safe bundle と challenger bundle の二系統管理、b035 依存 task の置換。 |
| leakage / overfitting risk | 高。public artifact の routing は private robustness を損ねやすい。 |

---

## 3. タスク単位の分析

### 3.1 すでに強く改善済みの task

| task | 状態 | 根拠 |
|---|---|---|
| task018 | public-zero repair 済み | `exp144` で b035 raw が LB gain。 |
| task023 | public-zero repair 済み | `exp152` で LB gain。 |
| task025 | 自前 rule repair 済み | `exp166` で high-cost でも 0->点 gain。 |
| task133 / task158 | public-zero repair 済み | `exp172` で 2 task repair。 |
| task285 | public-zero repair 済み | `exp178` で LB gain。 |
| task300 | rule lowering 済み | `exp234` で cost `77546 -> 52653`、LB +0.39。 |
| task060 / task226 / task123 | graph surgery delta 採用済み | `exp297` で fresh bypass delta。 |

### 3.2 まだ Public LB 上の未修復候補がありそうな task

| task / group | 状態 | 次の扱い |
|---|---|---|
| 未 probe task 全般 | 残 gap の一部がまだ存在する可能性 | 高 point 順だけではなく、source risk と subset-sum 一意性で probe group を再設計。 |
| task101 / task243 | fail-stub 上は zero らしく見えたが repair no gain | 同じ source での追加 repair は停止。別仮説が出るまで保留。 |
| task187 | 以前は zero と見えたが current lineage では alive らしい | high-cost repair は採用しない。自前低 cost 化できるなら防御的に再検討。 |

### 3.3 validation では promising だが LB に出ていない task

| task | promising な理由 | 障害 |
|---|---|---|
| task185 | `exp331` で `267_pass_0_fail`、cost `59584 -> 48889` | Public LB `6009.15` で採用済み。 |
| task037 | Python rule と dense ONNX が `266_pass_0_fail` | dense lowering cost `6633317`。compact representation が必要。 |
| task251 | Python rule `266_pass_0_fail`、GridSample cost `1800` 確認 | 新色 color1 を生成する mask/recolor が必要。 |
| task048 | GridSample priority queue に残る | runtime-stopper / rule lowering 未解決。 |

### 3.4 ONNX cost が障害になっている task

| task | 問題 |
|---|---|
| task037 | Slice/Pad shift-stack の full-grid intermediates が巨大。 |
| task187 | flood-fill unroll は correctness-first では通るが cost が高い。 |
| task251 | component mask generation が full-grid flood/reachability になると cost wall。 |
| task185 | `exp330` では Tile index が障害だったが、`exp331` の `Gather(axis=2/3)` で解消済み。 |

### 3.5 実装可能性はあるが工数が重い task

| task / family | 理由 |
|---|---|
| L4 shape/crop family | GridSample / Gather index map が成立すれば大きいが、過去の crop lowering は失敗が多い。 |
| L2 sparse fill family | task020 型の rule decomposition が必要。1 task ごとの診断工数が重い。 |
| task251 | closed component mask の低 cost 化が難しい。 |
| task037 | sparse diagonal table 化が必要で、単純実装では爆発する。 |

---

## 4. 失敗実験からの学び

### 4.1 仮説が間違っていた

- strict seed の -352 gap を subset-sum だけで直接特定できる、という仮説は弱かった。probe では alive が多く、算術は候補生成に留めるべき。
- public artifact を丸ごと採用すれば LB が上がる、という仮説は `exp130` で崩壊した。
- `task101/task243` は repair すれば gain する、という仮説は `exp321` で否定された。

### 4.2 実装表現が高コストすぎた

- flood-fill / component unroll / full-grid Where / Tile / GatherND は correctness が出ても cost が悪化しやすい。
- `task037` の dense shift-stack は full pass でも cost `6633317` で使用不可。
- `exp204` 系の巨大 row/col template は task185 で一度失敗済み。`exp330` では compact 化しても +200 cost が残ったが、`exp331` の `Gather(axis=2/3)` で解消した。

### 4.3 validation と public distribution がずれた

- full-arc pass の public artifact raw が Public LB で点にならない例がある。
- public-zero repair で b035 raw は効いたが、franksunp raw は効かないケースがある。
- したがって full-arc validation は必要条件であって、source 採用の十分条件ではない。

### 4.4 probe の情報価値はあった

- fail-stub probe は expected drop と実 LB が丸めレベルで一致し、task alive/zero の診断に強い。
- `exp323` / `exp325` の all-alive も「その suspect source の収率が下がった」という重要な情報であり、無駄ではない。
- 採点待ち中に次の audit / lowering を進める運用は、06-10 の LB gain に直結した。

### 4.5 今後再利用できる部品

- bundle ledger / lineage exclusion / quarantine。
- full-arc gated graph surgery replay。
- task185 dynamic bg detector と compact `Gather(axis=2/3)` lattice extraction。
- GridSample official cost probe。
- fail-stub probe builder と expected drop 計算。
- task-specific notes / Daily Log の PDCA フォーマット。

---

## 5. 中長期戦略

### Phase A: 1日以内にやるべき高期待値実験

| 項目 | 内容 |
|---|---|
| 目的 | current best `exp297` を壊さず、短時間で score-direct な候補を作る。 |
| 優先順位 | 1. task037/task251 の最小 compact probe、2. 強い public-zero suspect がある場合のみ fresh probe、3. task185 の compact Gather pattern 横展開。 |
| 具体的 experiment 案 | `exp332_task037_sparse_diag_probe`, `exp333_task251_mask_color1_minimal_probe`, `exp334_public_zero_probe_uniqueness_d` |
| 成功条件 | full-arc pass かつ cost 改善、または probe で新しい recoverable zero を特定。 |
| 中止条件 | fresh probe がさらに all-alive、compact diagonal / component mask が cost wall、task185 pattern の横展開が成立しない。 |
| 想定 LB gain | public-zero は 1 task +12〜15、compact rule は +2〜5。task185 は exp331 で +0.19 実現済み。 |
| 主なリスク | 小さい gain に提出枠を使いすぎること。Public LB feedback への過剰適合。 |

Phase A の焦点は「目先の submit 価値」と「次の lane 選択の情報価値」を両立することである。task185 は exp331 で採用済みなので、次は task037/task251 に compact lowering の成功パターンを移す。

### Phase B: 2-4日で積み上げるべき実験群

| 項目 | 内容 |
|---|---|
| 目的 | solved-rule task を cost-aware lowering で複数換金し、LB6100-6250 帯への足場を作る。 |
| 優先順位 | 1. GridSample/compact lowering の実 task 接続、2. public-zero probe の再設計、3. b035 依存 repair の防御的置換。 |
| 具体的 experiment 案 | task037 sparse diagonal、task251 mask+color1、task048 runtime-stopper audit、b035 repaired task の self-rule mining。 |
| 成功条件 | 2-4日で 2 task 以上の full-arc cost gain、または recoverable public-zero を 3 task 以上特定。 |
| 中止条件 | 4 target 中 3 target で cost wall、または 5 probe 連続で all-alive。 |
| 想定 LB gain | +15〜80。public-zero が再び見つかれば上振れ。 |
| 主なリスク | 低 cost 表現探索が長引き tooling 化へ逃げること。source raw 依存を増やすこと。 |

Phase B では、1 task ごとの rule lowering を「診断 -> 最小 ONNX -> cost gate -> submit 判断」まで短く回す。broad scan は使わず、task-specific に限定する。

### Phase C: 1週間以上の中期投資候補

| 項目 | 内容 |
|---|---|
| 目的 | LB7000+ に必要な family-level 圧縮へ移行する。 |
| 優先順位 | 1. L4 shape/crop、2. L2 sparse fill、3. signature lookup 族の rule 圧縮。 |
| 具体的 experiment 案 | GridSample/Gather index map の L4 15 task wave、task020 型 decomposition pipeline の L2 5 task wave、safe/challenger bundle 二系統管理。 |
| 成功条件 | 1週間で 10 task 以上の cost band 改善、または family 横展開で +100 以上の local delta 見込み。 |
| 中止条件 | L4 15 task 中 2 task 未満、L2 5 task 中 1 task 未満しか成立しない場合は LB7000 計画を public-zero + solved-rule 現実路線へ縮小。 |
| 想定 LB gain | +100〜500。LB7000 にはこの規模の family 成功が必要。 |
| 主なリスク | lookup/table 化による leakage、private collapse、長期投資の no gain 化。 |

Phase C は LB7000+ の本丸だが、現時点でいきなり全 family へ広げるのは早い。Phase A/B で compact lowering の成功パターンを 2-3 件作ってから横展開する。

---

## 6. 意思決定ルール

### 6.1 Kaggle submit する条件

- 診断 probe: fail-stub group の expected drop が明確で、subset-sum 解読価値が高い場合は submit する。
- Public-zero repair: full-arc pass し、expected LB gain が 1 task 分以上ある場合は cost 不問で submit してよい。
- Cost improvement bundle: full-arc pass、official score cost 改善、fresh local delta が原則 `+0.05` 以上なら submit 候補。
- 小数点表示に埋もれる micro delta は、他の bundle とまとめる。

### 6.2 local probe で止める条件

- subgraph の correctness / cost だけを測る段階。
- full-arc pass していない。
- cost 改善が baseline 未満、または gain が `+0.05` 未満で単独提出価値が薄い。
- public artifact source の private robustness が不明で、直接採用ではなく audit が目的。

### 6.3 系統ごと捨てる条件

- 5 実験連続で full-arc cost gain が 0。
- 同じ source / ranking の public-zero probe が 3 回以上 all-alive。
- lowering が full pass しても cost が baseline の 2 倍以上で、compact 化の具体案がない。
- tooling-only の追加で score-direct candidate が生まれない。

### 6.4 pending submission 中に進めるべきこと

- 次 probe group の選定と expected drop 表の作成。
- repair source audit。
- independent な task-specific rule audit。
- notes / Daily_Log / LB_Tracking の更新。
- scoring 結果に依存しない local cost probe。

pending score に依存する次手、たとえば「この probe が zero を出したらその task を repair する」は、結果が出るまで実装を分岐させない。

### 6.5 Public LB に過剰適合しない制約

- public artifact raw は task 単位で採用し、source 丸ごと blend はしない。
- b035 依存 repair task は最終 bundle で risk として明示し、可能なら自前 rule に置換する。
- final は safe bundle と challenger bundle を分ける。
- Public LB probe の結果だけで rule を作らず、必ず input-output rule と full-arc validation を通す。
- `data/raw` は変更しない。submission / model artifact / raw data は commit しない。

---

## 7. 次の 5 実験

### 7.1 `exp331_task185_gather_axis_cost_shave`

| 観点 | 内容 |
|---|---|
| 仮説 | `exp330` の `GatherElements` + Tile index を ONNX `Gather` axis に置換すれば、大きな index tensor を消して baseline `59584` 未満にできる。 |
| 実装内容 | `best_row/best_col -> start/spacing -> row_vec/col_vec` を作り、`Gather(axis=2)` と `Gather(axis=3)` で 4x4 lattice を抽出。default-bg / Pad はまず `exp330` と同じにする。 |
| 期待 gain | 実績: cost `59584 -> 48889`、Public LB `6009.15`、delta `+0.19`。compact lowering 原則として価値がある。 |
| 所要時間 | 20-40分。 |
| 成功条件 | `267_pass_0_fail` かつ cost `<59584`。 |
| submit 判断 | delta `+0.05` 以上なら submit 候補。微小なら bundle 待ち。 |
| notes 観点 | Tile index memory の削減量、Pad/default-bg の残 cost、shape inference の安定性。 |

### 7.2 `exp332_task037_sparse_diagonal_changed_cells`

| 観点 | 内容 |
|---|---|
| 仮説 | task037 は changed cells が少ないため、full-grid shift-stack ではなく per-diagonal / sparse changed-cell construction なら cost を抑えられる。 |
| 実装内容 | endpoint distance 1-5 の diagonal pairs を小 index table 化し、候補 cell だけ更新する ScatterND または Gather+mask を試す。 |
| 期待 gain | cost < baseline `63726` なら最大 +4 台。 |
| 所要時間 | 1-2時間。 |
| 成功条件 | `266_pass_0_fail` かつ baseline 未満。 |
| submit 判断 | full pass + cost gain なら submit 候補。 |
| notes 観点 | int64 index cost、changed-cell 数、ScatterND の memory、dense fallback の有無。 |

### 7.3 `exp333_task251_mask_color1_minimal_probe`

| 観点 | 内容 |
|---|---|
| 仮説 | task251 は pure GridSample では不可だが、closed zero-component mask と cheap color1 injection を分ければ cost を抑えられる。 |
| 実装内容 | まず component mask の最小表現を探索し、color1 one-hot を小 subgraph で合成する。GridSample は必要な場合のみ使う。 |
| 期待 gain | correct candidate が cost 10000 以下なら +2〜4。 |
| 所要時間 | 2-4時間。 |
| 成功条件 | Python rule と ONNX output が一致し、cost が baseline `100580` より十分低い。 |
| submit 判断 | full pass + fresh delta `+0.05` 以上なら submit。 |
| notes 観点 | 新色生成の表現、component mask の汎化、flood-fill 的 unroll に落ちていないか。 |

### 7.4 `exp334_public_zero_probe_uniqueness_d`

| 観点 | 内容 |
|---|---|
| 仮説 | 高 point 順だけでなく、subset-sum 一意性と source-risk を最適化した group なら、まだ未特定 public-zero を効率よく検出できる。 |
| 実装内容 | 未 probe task から 12-16 件を選び、全 subset sum の衝突が少ない group を作る。fail-stub zip を提出し、observed drop を解読する。 |
| 期待 gain | 直接 gain は 0。repair につながれば 1 task +12〜15。 |
| 所要時間 | 30-60分 + 採点待ち。 |
| 成功条件 | zero 候補が一意または少数に絞れる。 |
| submit 判断 | 診断価値が高ければ submit。直近 all-alive が続いたため、group 設計が弱い場合は submit しない。 |
| notes 観点 | group 選定理由、expected all-alive LB、missing drop、subset-sum ambiguity。 |

### 7.5 `exp335_b035_repair_self_rule_audit`

| 観点 | 内容 |
|---|---|
| 仮説 | b035 raw に依存している repair task を自前 rule へ置換できれば、Public LB best の private robustness が上がる。 |
| 実装内容 | task018/023/133/158/285 などの repaired task を対象に、input-output 差分、source raw 構造、train/test/arc-gen パターンを監査する。まず 1 task を選び rule hypothesis を作る。 |
| 期待 gain | 直接 gain は 0、防御的価値。低 cost 化できれば微小 gain。 |
| 所要時間 | 2-4時間。 |
| 成功条件 | 1 task で説明可能 rule または source-independent candidate が見つかる。 |
| submit 判断 | 同等以上 cost かつ full pass なら challenger ではなく safe bundle 側に採用検討。 |
| notes 観点 | leakage risk、public artifact 依存の除去度、private robustness 仮説。 |

---

## 8. 投資判断の結論

短期では、`task185` の narrow cost shave は exp331 で成功し、Public LB `6009.15` に転写した。次は深追いせず `task037` / `task251` の compact lowering に移る。Public-zero repair は依然として最大の実績レーンだが、直近の高 point probe は収率が落ちたため、次は group 設計の質を上げた probe に限定する。

中期では、LB7000+ には public-zero と surgery だけでは足りない。GridSample / compact Gather / sparse update のような低 cost 表現を 2-3 task で実証し、それを L4 shape/crop と L2 sparse fill family へ横展開する必要がある。

最重要の制約は、現 best `exp297` を壊さないこと、public artifact raw 依存を増やしすぎないこと、そして tooling-only に戻らないことである。以後の実験は「full-arc correctness」「official cost」「fresh delta」「leakage/private risk」を同時に記録し、submit するか local probe で止めるかを毎回明示する。
