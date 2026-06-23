# exp338 — P0-1 local↔LB 採点校正（sentinel解消）

/ 目的: 計画書 §2.2 / §5 Phase0 / §8 P0-1。union zip のローカル公式再採点を Kaggle LB に近づけ、
sentinel を解消して自律エンジンの単調安全弁を「真の floor」に対して機能させる。

## 仮説（計画書）
7タスク(45,127,135,146,149,240,384)の **ConvTranspose 負pads** が `onnx.shape_inference` で
採点失敗 → sentinel(cost=1e9, pts=1.0)。これを解消すれば baseline がローカルで 7117.01±1 を再現する。

## 実測で確定した事実（diag/probe で検証）
1. **失敗点は `neurogolf_utils.calculate_memory` 先頭の `onnx.checker.check_model(full_check=True)`**
   （内部で strict shape inference 実行）。`infer_shapes(strict_mode=True)` も同じ
   `[ShapeInferenceError] Attribute pads must not contain negative values` を投げる。
2. **ConvTranspose だけでなく Conv も対象**（task135/146 は plain Conv の負pads）。計画書の
   「7 ConvTranspose」は不正確。トリガは **Conv/ConvTranspose の負pads** 全般。
3. `infer_shapes(strict_mode=False)` は7件すべて成功。**ORT は7件すべて load+run 成功**し
   正しい (1,10,30,30) を出力 → モデルは実行可能で妥当。strict チェックだけが過剰却下している。
4. ローカル onnx==1.22.0 が strict 化しており、Kaggle の onnx ビルドは負pads を許容して採点していると推定。
5. **7件は高コストではなく安価**(cost 100〜1993, pts 17.4〜20.4)。計画書の「93点≒7 sentinel(≈14pt/件)」
   は誤り。実際は7件で **約125点**相当。

## 3案比較と採用
| 案 | 内容 | 判定 |
|---|---|---|
| **A: ORT実行(trace)ベース採点【採用】** | memory を ORT profiling trace の output_type_shape から算出、params は公式 calculate_params。shape inference 非依存。 | ◎ 7件すべて採点可(149含む)、堅牢、改変なし |
| B: shape_inference 迂回(strict=False) | 公式 calculate_memory を strict=False で実行 | △ memory を過小計上(045で約-320)、task149 は依然 None(空名ノードで value_info 欠落) |
| C: 負pads正規化 | 7 artifact を書換えて負pads除去 | ✗ artifact の cost が変わり **校正を破壊**(Kaggle は元 artifact を採点)。採用不可 |

採用=**案A**。`neurogolf_calib.score_model_calibrated`:
公式 score_model を先に実行 → 393件は **byte一致**で不変。負pads系のエラー時のみ `_score_via_trace`
にフォールバック。公式 `neurogolf_utils.py` は無改変(reference保全)。

## 結果
- calibrated baseline = **7147.94 / sentinel 0件**。393件は旧公式と cost 完全一致(393/393)。
- 7件: 045=1993, 127=408, 135=200, 146=495, 149=100, 240=1475, 384=759 (pts 17.4〜20.4)。
- `baseline_ledger.json` を calibrated 版に差替(旧版は `*_official_sentinels.bak.json` に退避)。
- climb 連携: build_baseline=calibrated化 / sweep で7件を **freeze**(skip) / 安全弁は解除(start>floor)。
- climb smoke (`--once --max-tasks 2`) 正常終了、改善0(現7 builder では2最難タスクに不適合=想定内)。

## 重要: 計画書前提の崩れ（要協議）
**sentinel を正直に採点しても total=7147.94 で、Kaggle 7117.01 を約 +31 上回る**。
→ §8 完了条件 total∈[7116,7118] は **到達不能**。差は7件だけでなく、ローカル onnx と Kaggle onnx の
版差による広域の僅差を含むと推定。**絶対校正は提出して観測する以外に確定できない**(=P1-2 の役割)。

## freeze の根拠（安全）
7件の baseline cost は **trace法**、候補は **公式法**で測られ比較が非整合。かつ Kaggle 上の真コスト未校正で、
ローカルで安い標準op置換が Kaggle で悪化する恐れ。→ P1-2 で LB 校正するまで **改善対象から除外**。

## leakage / overfit リスク
- 校正は採点器のみ変更、データ・artifact 不変 → leakage なし。
- freeze により7件の過学習的置換リスクを遮断。
- 393件は不変 → 既存単調性に影響なし。

## 再現
`python exp338_p0_calibration/run_calib_baseline.py`(非破壊, 別ledger出力) /
`python exp338_p0_calibration/verify_climb_phase0.py`(climb連携の受入確認)。
中核モジュール: `workspace/experiments/neurogolf_calib.py`。
