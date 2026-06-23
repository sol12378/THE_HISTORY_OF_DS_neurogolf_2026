# exp340 — Phase 2 コスト解剖 + 一般レバー3種の検証（全て dead-end を確定）

/ 目的: Phase2(桁落とし=本丸)着手。「安全に一段ずつ」で、まずコストの在処を実測し、
適用すべきレバーを確定 → 一般変換は full-arc gate を通して安全に検証。

## Step 0: コスト解剖（top-30 高コストタスク）
- cost = memory(bytes, ORT trace) + params(**elements**, dtype非依存)。
- **top-30 は全て memory-bound(90–100%)**。params寄与は小。
  → dtype最小化が理屈上の本命。だが下記で CPU では効かないと判明。
- op構成は And/Or/Where/Cast/Equal/Greater 等の bool/比較中心 = **input依存の本体計算**。
- 上位: 187=78309(99% mem,78n), 233=75476(481n), 18=71778(953n), 286=69848, 209, 205…

## Step 1: float16 変換（keep I/O float32, 内部fp16）— **dead(CPUで逆効果)**
- 手実装(onnxconverter_common はlock整合のため導入不可)。init/const/Cast-to を fp16、
  境界は Cast で橋渡し、value_info はクリアして再推論。
- 結果: task255 は **full-arc pass(265/0)** したが **cost 45044→68228 と増加**。
  原因 = **CPU ORT は fp16 ネイティブkernel が無く内部で fp32 へ up-cast**し、
  scorer は **実行trace(=fp32)** を計上。さらに内部cast tensorが増えて memory増。
- 他5件は strict shape-inference で reject。→ **gate が全件正しく拒否(無害)**。
- 教訓: **scorerは実行された実テンソルbytesを測る。CPUで効くのは bool/int ネイティブ型のみ。**
  fp16/uint8 の一括dtype変換は CPU では cost を下げない。

## Step 2: 定数畳み込み(input非依存中間を initializer へbake)— **dead(yield≒0)**
- bake で float32定数(N elem): memory -4N, params +N = **cost -3N** の理屈。
- 実測(top14, 入力3種で中間の input非依存性を判定): **13/14 が const_f32_elems=0**。
  union artifact は定数を既に initializer 化済で、float32中間は全て **input依存の本体**。
  唯一 task255 が 1370 elem(≒-4110 cost ≒ +0.1pt)。→ 実質無し。

## Step 3: prune(dead node除去) — **dead(既に最小)**
- P1 sweep(exp339)で全393に prune 適用済 → 0改善。

## 結論（Phase2の現実）
**一般変換(fp16/uint8一括, const-fold, prune)は全て empirically dead。**
公開7117 union は per-task で既に十分golf済。実利得は **bespoke per-task 再lowering**
(P2-4)＝各タスクの幾何/規則を逆解析しより安いグラフを再構築、または emitter拡張(P2-2)で
新しい安価primitiveを供給、しか道がない。例: task349(16n,41376)は ConvTranspose×2 が
beam/halo幾何を符号化した大float32中間でコスト集中 → 等価な安価構築には幾何の逆解析が必要。

各 bespoke は数時間規模・利得は1タスク ~+1pt(cost桁落ち時)。投資判断はユーザーへ。

## 安全性
- 全変換は `evaluate_candidate`(static+full-arc(-1)+rescore, cost厳密減のみ採択)で gate。
- 本exp は探索のみ・**提出なし・union/baseline 無変更**。leakage/overfit なし。

## 再現
`probe_cost_anatomy.py`(memory/params解剖) / `test_fp16.py`(fp16 gate検証) /
`probe_constfold.py`(定数yield計測)。いずれも `workspace/experiments` から実行。
