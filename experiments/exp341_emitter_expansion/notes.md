# exp341 — P2-2 emitter拡張(基盤整備) + P2-1 gate緩和 + GridSample価値検証

/ 目的: ユーザー選択「emitter拡張を先に基盤整備」。neurogolf_farm の emitter を3→拡張し、
cost_extractor gate を size-conditional 化。さらに新capability(GridSample)が union を
破れるかを scan で検証。

## 実装(P2-2 emitter / P2-1 gate)
`neurogolf_farm/emitter.py` に5 primitive の lowering を追加:
- **GRID_SAMPLE**(opset16, 単一GridSample+座標grid) — phase1 に無い新capability。**cost 1800**(exp324再現)。
- **RECOLOR_DIRECT**(単一 Gather, 中間ゼロ=output除外) — **cost 10**。bijective recolor 用。
- **COMPUTED_SLICE_PAD**(offline-baked static の Slice+Pad) — valid。
  ※ 真の input依存 dynamic bounds は `infer_static_ok` が dynamic shape を reject するため不可能。
    static 制約下では STATIC_SLICE_PAD と等価(=重要な知見)。
- **SMALL_LOCAL_MASK**(Where) — lower は valid だが full Where は cost 18000(=fixed_mask_where 同等)。
  真に「small」にするには crop+edit+pad の bounded 構築が必要(未実装、要追補)。
- **RECOLOR_CAST**(Gather+Cast uint8) — **無効**: make_model の output=float32 契約と uint8 出力が衝突し
  gate reject。かつ output は memory 計上外なので uint8 化の利得ゼロ → RECOLOR_DIRECT が支配。deprecated。

`neurogolf_farm/cost_extractor.py` (P2-1): FULL_GRID_COMPOSITION/CONNECTIVITY_UNROLL/SPARSE_WRITEBACK の
無条件 hard_reject を **size-conditional** 化(full-grid出力時のみ reject、小領域は許可)。
smoke 確認: SPARSE_WRITEBACK 3x3→allowed / full-grid→rejected。

## 価値検証: GridSample inducer scan(全393)
pixel-permutation 検出(output[r,c]=input[f(r,c)], content非依存, 全例一致)→ grid fit → 単一GridSample
→ full-arc gate。結果:
- **22タスクが permutation-like で fit & full-arc pass(正しい)**。
- だが **全22で union cost < GridSample 1800** → **wins=0**。
  - union が解けるタスクは単純(flip/rot=Gather cost30, identity cost0-5)で 1800 より遥かに安い。
  - union cost>1800 の高コスト群は **純permutationでない**(新色生成・非copy変換)→ inducer が fit しない。
- GridSample の cost下限は full-grid grid=30×30×2=1800。小出力→Pad は中間追加で逆に高coст。

## 結論(Phase2 一般アプローチ全滅を確定)
exp340(dtype/const-fold/prune 全dead)+ exp341(emitter/GridSample 0win)で、
**安全な一般変換・汎用primitiveでは union(7117)を破れない**ことを実証。公開fieldが7117で停滞する理由。
残る道は **bespoke per-task 再lowering(P2-4)** のみ — 高コスト非permutationタスク(187/233/18…)の
変換を逆解析し等価で安いグラフを再構築。各数時間・利得不確実(過去 exp317/327/332 等も union 未達)。

## 安全性・統合方針
- emitter変更は farm モジュール内。**climb の候補生成には未配線**(0win のため配線すると sweep が
  無駄に重くなるだけ)。→ 自律ループ・floor・提出に影響なし(all-green 維持)。
- 将来 inducer(P3)が GridSample/recolor の用途を見つけた時点で配線する。
- 全 scan は gate 経由・提出なし・union/baseline 無変更。leakage/overfit なし。

## 再現
`smoke_emitter.py`(primitive lowering + gate緩和の検証) / `gridsample_scan.py`(全393 scan)。
