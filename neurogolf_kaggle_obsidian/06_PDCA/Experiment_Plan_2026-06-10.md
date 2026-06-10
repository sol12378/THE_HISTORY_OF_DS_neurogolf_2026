# 実験計画 2026-06-10(締切まで35日)

状態: 提案(ユーザー承認待ち)。承認後にDecision_Logへ採用判断を記録する。

## 1. 前提となる客観的事実

### コンペ構造

- 400 ARC taskへtaskごとに最小ONNXを提出。task得点は functional correct なら `25 - ln(cost)`(下限1)、不正解taskは0点。`cost = params + memory bytes`。理論上限 10000。
- 締切 2026-07-15 23:59(残り35日)。1,774チーム、賞金 $50,000。
- スコアモデルはローカルで厳密再現済み: exp066のtask020 `90133 -> 73080` はln比 `+0.2097` で、LB delta `+0.21` と完全一致。

### 現在地

| 項目 | 値 | 根拠 |
|---|---:|---|
| submit-safe best LB | 5930.40 | exp_b025 ref 53418067 |
| submit-safe working local | 6282.965236 | exp127(+0.153が未提出) |
| local/LB較正gap | -352.34〜-352.41(6提出で一定) | LB_Tracking |
| lookup系collapse | -3062.59 | exp041 ref 53414511 |
| public CODE collapse | LB 1400.37 | exp130 ref 53450559 |
| 全400 cost<=600 floor projection | 7503.63 | exp053 |
| 全400 cost<=250 floor projection | 7833.83 | exp053 |

### -352.34 gapの解釈(本計画の核)

1. exp001(task087単独提出)はlocal 14.507 / LB 14.50で**gapほぼゼロ** → hidden側のcost計測はlocalと一致し、正解taskのスコアは厳密に転写される。
2. micro deltaは6回連続でlocal/LB 1:1転写(+0.21, +0.17, +0.13, +0.03, +0.13)。
3. taskスコアはall-or-nothing(不正解=0点)。
4. よって-352.34の源泉は「strict seed内の固定的な一部taskがprivate benchmark suiteでfunctional failしている」以外に整合的説明がない。失敗task数の推定: 352.34 ÷ 平均14.7点 ≈ **24 task前後**。
5. この+352はcost圧縮ゼロでも回収できる確定的な伸びしろであり、未着手。

### 効いた手法/効かなかった手法(証拠)

効いた:
- 明示rule → ONNX lowering(exp066 task020、唯一のrule→LB実証 +0.21)。
- full-arc gated graph surgery / Seddik-style post-pass(LB較正用micro delta、ただしexp124-127で+0.003/passまで逓減)。

効かなかった(全てguardrail化済み):
- naive lowering全般: flood-fill unroll、full-grid Where/Tile、ScatterND、dynamic MatMul、full-grid GatherND、Conv visibility(exp019-021, 024-025, 044, b011, b028)。
- 汎用optimizer(exp009, exp099)、broad anchor/template scan(exp104-106)、one-node template(exp096-098)、public blend直接採用(exp130)。

蓄積資産:
- 解決済みrule 8件(task020✓済 / 037 / 251 / 085 / 366 / 365 / 185 / 048、ほぼ全てfull-arc 266/266)。
- farm pipeline稼働確認済み(exp129-134): IR→emit→validate→ledger→bundle。**supplier(候補生成器)が空**なのが現ボトルネック。
- cost物理: fp32 one-hot 30x30中間1個=36000 bytes ≈ そのtaskは最大14.5点。one-node data movementはほぼ無料(identity 0, flip 4, channel gather 10; exp094)。task085 artifactはfp16内部(exp_b032)→ 上位golferはdtype縮小を使っている形跡。

## 2. 戦略算術と目標バンド

LB 5930.40から:

| レーン | 上限期待値 | 確度 | 根拠 |
|---|---:|---|---|
| A: private failure修復 | +352 | 高 | gap一定+exp001ゼロgap+all-or-nothing採点 |
| B: LB-verified public選別取込 | +0〜270 | 低 | LB6200+情報は未検証。exp008/exp130は失敗実績 |
| C: cost band圧縮(farm量産) | +300〜1550 | 低〜中 | 250 floor projection 7833。lowering cost wallが障壁。dtype laneは未検証 |

- A完全成功でLB ≈ local 6283。
- 7500には「Aに加えC側で+1200(≈250 taskを600級化)」が必要 → 35日では**ストレッチ目標**。
- 現実的目標バンド: **6280(A成功) → 6500-6800(C第1波) → 7500(ストレッチ)**。

## 3. フェーズ別実験計画

### Phase 0(即日)

- **P0-1 exp135: exp127 working bundle提出** — 未提出+0.153を提出しLB best更新と較正gap恒常性の再確認。受入: LB ≈ 5930.55。
- **P0-2: 提出予算確認** — 日次提出上限をKaggle UIで確認し、Phase Aのbisection probe枠を確保。最終提出2枠は常時「safe best + challenger」を選択。

### Phase A: private failure特定・修復(Day 1-7、最優先)

- **A-1 exp136: arc-gen供給源の確認とstress例増産** — 手元arc-gen(266例/task)が固定配布か生成器かを確認。生成器(ARC-GEN系)が動くならseed拡張で2000+例/taskを生成し、strict seed 400 artifactの失敗率を測定。family単位で20分ジョブに分割。受入: 失敗候補taskリスト(期待 ~24件)。
- **A-2 exp137: 較正算術クロスチェック** — 全400 taskの正確なofficial costからper-taskスコアを計算し、「失敗集合の合計 = 352.34〜352.41」を満たす部分集合をA-1候補・provenance(出所artifactのlookup様構造)・arc-gen coverage薄さと突合。
- **A-3 exp138-140: 提出bisection probe(必要時のみ)** — suspect群を意図的fail stub(極小不正解model)に差し替えた提出でgroup単位の通過スコア質量を測定。1 probe = 1提出。A-1/A-2で確度が出れば省略。予算: 最大3-6提出。
- **A-4 exp141+: 失敗task修復** — 失敗型(shape範囲・色数・edge case)を診断し、(a)別public sourceの同task full-arc valid model、(b)rule lowering、(c)頑健化surgeryの順で差し替え→micro delta提出で1 taskずつ確認。1 task修復 ≈ +14.7 LB。
- リスク: hidden suiteがarc-gen非互換ならA-1空振り→A-3に切替。gapの代替仮説(部分採点等)はA-3で判別可能。

### Phase B: LB-verified public sourceの検証(Day 1-10、並行・小予算)

- **B-1 exp142: exp_b039完了** — jonathanchan等のscanを完了し、per-task cost / full-arc validation / 構造監査(lookup様 MatMul/ScatterND署名照合は除外)をregistry化。
- **B-2: LB evidence確認** — 実LB 6200+の裏付けが取れたsourceのみ「LB-verified teacher」へ昇格。local claimは採用根拠にしない(exp130の教訓)。
- **B-3 exp143: 段階blend probe** — 昇格sourceがある場合のみ、~50 task単位の差し替え提出でprivate通過率を計測。collapse即検知・即rollback。
- 判断基準: B-1で全sourceがlookup様/full-arc failなら、Bは打ち切りCへ集中。

### Phase C: cost band圧縮の量産 — farm本稼働(Day 5-25、主力)

ボトルネックは「rule発見」ではなく「安いlowering表現」。supplierを3系統に絞る:

- **C-1 exp144: dtype/メモリ表現プローブ(最優先・20分)** — 同一graphでfp32/fp16/bool/uint8中間のofficial `score_network` costを系統計測。one-hotは0/1なのでbool/uint8計算+終端Castが原理的に可能(36000→9000 bytes、+1.39点/task級)。fp16半減でも+0.69点/task級。
  - 効果確認後 **C-1b exp145: 既存400 artifactへの自動fp16/bool化post-pass**(full-arc gate付き)。横断適用できれば+100〜270級の単発最大レーン。
  - 根拠: cost=memory bytes(公式)、task085のfp16内部実在(exp_b032)。リスク: ORT profilerのmemory計上がdtype比例でない可能性→先にプローブ。
- **C-2 exp146-148: 低cost artifact逆設計** — cost<=250の19実例から「動的処理を1-2 nodeで済ます」技法を抽出しfarm IRテンプレート化。優先: `one_node_gridsample`(9例; 座標計算をattribute/小tensorへ押し込む)、`gather_index_map`(8例)、`computed_slice_pad`(20例; exp100)。
- **C-3 exp149-155: 解決済みrule 7 taskの専用lowering** — task037/251/085/366/365/185/048へC-1/C-2の新表現を適用。1成功 = +2〜5点 + LB較正即提出。
- **C-4 exp156+: family横展開** — 確立した表現でL4 shape/crop(62 task, +279)→L2 sparse fill(65 task, +273)の順にfarm量産(exp054 taxonomy準拠)。既存pre-emission cost gateを維持。5実験ごとreview gate。

### Phase D: 統合・防御(Day 25-35)

- **D-1: best bundle統合union**(exp_b025方式)+full-arc再検証+提出。
- **D-2: 最終週は新規リスク凍結** — A-4型修復とmicro deltaのみ。最終2枠を確定。
- **D-3: 文書化・再現性確認。**

## 4. 週次タイムライン

| 週 | 期間 | 主作業 | 目標LB |
|---|---|---|---:|
| W1 | 6/10-6/16 | P0 + A-1〜A-4第1波 + B-1/B-2 + C-1プローブ | 6000-6150 |
| W2 | 6/17-6/23 | A-4継続 + C-1b横断post-pass + C-2/C-3 | 6280-6450 |
| W3 | 6/24-6/30 | C-3完遂 + C-4 L4第1波 | 6500-6700 |
| W4 | 7/1-7/8 | C-4 L2第2波 + D-1統合 | 6700-7000+ |
| W5 | 7/9-7/15 | D-2凍結・最終選択 | - |

## 5. 運用ルール

- 各実験は1目的・20分優先・`result.json`/`notes.md`必須(既存規約)。
- official-valid deltaは温存せず即提出して較正(Decision_Log既定方針)。
- worker agentは低reasoningで狭く: stress生成ジョブ、artifact構造監査、lowering cost probe、family横展開を分担。main agentがfull-arc gate/採否review。
- 5実験ごとにreview(既存運用継続)。
- leakage/overfitting: raw lookup構造は提出禁止、full-arc validationは必要条件であって十分条件ではない(exp041教訓)→新規表現は必ず小規模delta提出でprivate通過を確認してから横展開。

## 6. 反証可能性

- A仮説(~24 task private fail)はA-3 probeで直接反証可能。
- C-1仮説(dtype縮小でmemory cost比例減)は20分の`score_network`実測で即判定。
- B仮説(LB6200+ public source実在)はprobe提出1回で判定。
