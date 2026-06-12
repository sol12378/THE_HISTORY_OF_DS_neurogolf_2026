# 実験計画 2026-06-12(締切まで33日・LB1位を目指す)

状態: 提案。承認後に Decision_Log へ採用判断を記録する。

## 1. 現状把握(客観的事実)

- **Current public LB best: 6009.15**(exp331 = exp297 lineage + task185 compact `Gather(axis=2/3)` lowering)。expected値とLBは丸め一致しており、較正は完全に効いている。
- **LB 1位は 7710.67(CroDoc)**、上位10チームは 7430〜7710 帯。我々とのギャップは **約 1700 点**。直近(6/11-6/12)も上位は伸び続けており、量産システムを持つ前提で読む。
- 直近のレーン収率:
  - **public-zero repair(Phase A)**: exp323/325 が連続 all-alive で収率枯渇。残 suspect なし。
  - **full-arc bypass surgery**: 107 task 掃いて合計 +2.68。逓減で実質終了。
  - **compact dynamic lowering**: task185 で **+0.20 をLB実証**(exp331)、task300 で +0.39 実証(exp234)。唯一伸びているレーン。
- 未完了の宿題: `experiments/exp332_task037_cropped_diag_shift_lowering/run_exp332.py` が実行待ち。
- 解決済み rule 資産(lowering 待ち): task037 / 251 / 085 / 100 / 271 / 346 / 174(部分)/ 048 など。

## 2. 1位条件の算術(計画の核)

スコアは task ごとに `25 − ln(cost)`(下限1、不正解=0)。平均点換算:

| | 平均点/task | 含意される平均cost |
|---|---:|---:|
| 我々 6009 | 15.02 | ≈ 21,600 |
| 1位 7710 | 19.28 | ≈ **305** |

つまり1位は**ほぼ全400 taskを cost 数百のcompact ONNXにしている**。exp053 の floor projection(全task cost≤600で7503、≤250で7833)と完全に整合する。

> **結論: micro delta(surgery +0.1級)の積み上げでは構造的に届かない。「rule発見 → compact lowering」を1 taskあたり +3〜5点で量産する体制だけが1位ルート。**
> 必要量は約 350〜400 task 分。残り33日なら **1日 12 task ペース**が要求水準 → 手作業では不可能 → farm(compiler pipeline)の supplier 充填と worker agent 並列化が必須。

## 3. フェーズ別計画(2026-06-12 → 07-15)

### Phase 0: 即日(本日)

1. **exp332 実行**: `.\.venv\Scripts\python.exe experiments/exp332_task037_cropped_diag_shift_lowering/run_exp332.py`。full-arc + cost を確認し、gain なら即提出。受入: task037 cost < 63726。
2. **上位手法 intelligence(20分×2)**: Discussion / 公開 notebook を再監査。平均cost 300 を実現する表現(汎用 micro-program 群か、fp16/uint8 + 1〜3 node データ移動か)の手がかりを収集。

### Phase 1: compact lowering パターンの確立(Day 1–5)

目的: task185 で実証した「**動的index は rank-1 vector + `Gather(axis=2/3)`**」「**selector/detector は `ReduceSum`+`ArgMax` の cost 数百 subgraph**」を、**再利用可能な lowering primitive ライブラリ**に固める。

- 解決済み rule queue を順に lowering: task037(exp332)→ task251(closed-component mask + color1; mask生成の compact 化が課題)→ task100 / 271 / 346(bbox/count selector の compact 版)。
- 各 task の受入基準: full-arc all pass **かつ** cost < 現baseline。成功ごとに single-task delta を即提出してLB較正(既定方針)。
- 失敗パターン(flood-fill unroll、full-grid Where/Tile、ScatterND、Conv visibility)を避ける設計規約を primitive ライブラリ側に焼き込む。
- 受入: 5日で 4〜6 task 成功、primitive ライブラリ v1(`gather_axis_dynamic_index` / `argmax_selector` / `bbox_compact` / `narrow_dtype_intermediate` など 5〜8 個)。

### Phase 2: farm 量産体制の本稼働(Day 5–20、主力)

farm pipeline(IR→emit→validate→ledger→bundle)は exp129–134 で動作確認済み。**supplier が空**が現ボトルネック。

- **S-1 rule miner worker 並列化**: exp054 taxonomy(L4 shape/crop 62 task、L2 sparse fill 65 task、ほか)を family 単位で worker agent に分配。各 worker は「入力のみの Python rule で full-arc 全pass」を狭く探す。1 worker = 1 family バッチ、低 reasoning、20分上限。
- **S-2 lowering worker**: rule が見つかった task を Phase 1 primitive ライブラリで lowering。pre-emission cost gate(既存 guardrail)で高cost案を事前棄却。
- **S-3 main agent review**: full-arc gate、cost比較、bundle ledger 採否。10〜20 task たまるごとに bundle 提出(expected LB を必ず事前計算し、乖離即ロールバック)。
- KPI: **週 +300〜500 LB**。W2終了で 6400 前後、W3終了で 6900 前後が健全ライン。下回ったら Phase 1 primitive の表現力不足を疑い、cost≤250 実例19件の逆設計(GridSample / gather_index_map / computed_slice_pad)へ再投資。
- 並行小予算: dtype 縮小は「既存artifactへのpost-pass」ではなく(exp159–161で否定済み)、**新規 lowering 設計時に bool/uint8 中間をデフォルト化**する形で回収。

### Phase 3: 高難度残タスクと上位ギャップ(Day 20–30)

- rule が見つからない残り task(おそらく100〜150件)を cost 順に攻める。効くのは「部分rule + 既存artifact のハイブリッド」「artifact 構造の深い書き換え(dtype/グラフ再構成)」。
- 我々が 250 task を 600 級化できれば 7200〜7400 帯 = **賞金圏**。1位には残り全部の圧縮が必要で、ここの歩留まりが勝敗を決める。

### Phase 4: 防御と最終選択(Day 30–35)

- 新規リスク凍結。best bundle の full-arc 再検証、union 統合、再現性確認。
- **最終2枠は「private-safe 寄り(strict lineage)」+「current best(public-zero repair 込み challenger)」**。現 best lineage は exp_b035/franksunp 系 source を含み private robustness リスクが残る点を明示的に持ち越す。可能なら W4 で challenger の高リスク task を rule-lowering 産に順次置換して risk を削る。

## 4. 週次タイムライン

| 週 | 期間 | 主作業 | 目標LB |
|---|---|---|---:|
| W1 | 6/12-6/18 | P0 + Phase 1 primitive 確立 + intelligence | 6050-6150 |
| W2 | 6/19-6/25 | Phase 2 farm 本稼働第1波(L4 shape/crop) | 6400-6600 |
| W3 | 6/26-7/02 | farm 第2波(L2 sparse fill)+ dtype 標準化 | 6900-7100 |
| W4 | 7/03-7/09 | Phase 3 残タスク圧縮 + D-1 統合 | 7200-7400 |
| W5 | 7/10-7/15 | Phase 4 凍結・最終選択 | ストレッチ 7500+ |

## 5. 運用ルール(継続+変更点)

- 1実験1目的・20分優先・`result.json`/`notes.md`・5実験ごと review は継続。
- **変更点1**: 提出予算は probe ではなく **bundle 較正提出**に再配分(public-zero probe は強い新 suspect source が出ない限り停止)。
- **変更点2**: surgery / tooling-only 実験(exp266–314 型)は原則停止。**すべての実験を「task X の cost を Y→Z にする」形に強制**。
- worker agent は低 reasoning で狭く: rule miner(family バッチ)、lowering cost probe、artifact 構造監査を分担。main agent が full-arc gate / 採否 review。
- leakage/overfitting: raw lookup 構造は提出禁止、full-arc validation は必要条件であって十分条件ではない(exp041教訓)→ 新規表現は必ず小規模 delta 提出で private 通過を確認してから横展開。

## 6. 反証可能性・判断点

- 「1日12 task ペースが可能か」は Phase 2 の最初の5日で直接判定できる。
- **W2 末の判断点**: 週 +150 LB 未満なら、1位目標を賞金圏(7200+)目標に修正する。
- C-1 仮説(dtype 縮小で memory cost 比例減)は新規 lowering 設計時の `score_network` 実測で随時判定。

## 7. 次アクション

**exp332 の実行**(task037、解決済み rule の compact lowering、Phase 1 の先頭)。
