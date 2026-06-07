# onnx-tool 実測校正レポート（確定版）

`probe_onnx_tool.py` と `test_all.py` で本物の onnx-tool を実行して得た**実測値**。
これまでの概算（要素数×dtype）を絶対値に校正した結果。

## 最重要の確定事実

### 1. memory項は「出力テンソルのバイト」のみ計上（入力は計上されない）

これが最大の発見。Transpose uint8 10×10 の memory = 100バイト（=出力10×10×1のみ）。
入力テンソルは memory に含まれない。

**含意**: 色置換で入力を int32 にしても memory コストは増えない。だから
`recolor_direct`（int32入力・Gather単発・中間ゼロ）が最安になる。実測で：

| 色置換の実装 | 中間テンソル | 実測cost | 備考 |
|---|---|---|---|
| recolor_direct（int32入力） | 0個 | **44** | ★最安。入力int32のバイトは計上されない |
| recolor_cast（uint8入力+Cast） | 1個(Cast出力) | 140 | Cast中間がmemory+100 |

→ **色置換は int32 入力の Gather 単発が正解**。Castを足すのは損。

### 2. dtype は memory に 8倍効く（uint8 vs int64）

Transpose 10×10 実測： uint8 = 100B、int64 = 800B。**uint8化でmemoryが1/8**。
出力テンソルを uint8 に保つことが cost 削減の核心と実証された。

### 3. FREE op は MAC=0（全確認）

Transpose / Slice / Tile / Concat / Pad / Gather / **Where** すべて MAC=0。
特に **Where が MAC=0** は重要：条件選択をMACゼロで行える。

### 4. CHEAP op は MAC=出力要素数

Add / Mul / And / Or / Xor / Equal / Greater = 出力要素数ぶんのMAC（10×10で100）。
ただし **Where だけは例外で MAC=0**。
→ 論理合成は「Equal/Xor等でマスク生成（MACあり）→ Where で適用（MAC=0）」が定石。

### 5. Slice/Tile/Pad/Gather の定数は params に計上

Slice params=6（starts/ends/axes等の定数要素数）。小さいが計上される。
Gather LUT params=10（色LUT）。これらは cost に直接加算される。

## 全テンプレート実測コスト一覧（4×6グリッド=24セル）

| テンプレート | MAC | memory | params | cost | score | 帯 |
|---|---|---|---|---|---|---|
| identity | 0 | 24 | 0 | 24 | 21.82 | <250 |
| transpose | 0 | 24 | 0 | 24 | 21.82 | <250 |
| recolor_direct | 0 | 34 | 10 | 44 | 21.22 | <250 |
| flip_lr | 0 | 56 | 4 | 60 | 20.91 | <250 |
| crop | 0 | 54 | 6 | 60 | 20.91 | <250 |
| recolor_where | 24 | 48 | 0 | 72 | 20.72 | <250 |
| rot90_ccw | 0 | 80 | 4 | 84 | 20.57 | <250 |
| pad | 0 | 80 | 4 | 84 | 20.57 | <250 |
| halves_xor | 12 | 72 | 0 | 84 | 20.57 | <250 |
| flip_both_fused | 0 | 88 | 8 | 96 | 20.44 | <250 |
| rot180 | 0 | 88 | 8 | 96 | 20.44 | <250 |
| crop_then_flip | 0 | 96 | 10 | 106 | 20.34 | <250 |
| mirror_tile_h | 0 | 104 | 4 | 108 | 20.32 | <250 |
| tile2x2 | 0 | 112 | 2 | 114 | 20.26 | <250 |
| two_grids_and | 24 | 96 | 0 | 120 | 20.21 | <250 |
| recolor_cast | 0 | 130 | 10 | 140 | 20.06 | <250 |
| rot90_recolor | 0 | 258 | 14 | 272 | 19.39 | ★250-600 |

## 解釈Aの予算設計への校正

4×6（24セル）では大半が cost 100 前後。**10×10（100セル）にスケールすると約4倍**になり、
多くが 250〜600 帯に自然に入る。グリッドサイズ別の memory（uint8・単一出力）：

| グリッド | 出力1枚のmemory | 2ノード | 3ノード |
|---|---|---|---|
| 4×6 (24) | 24B | 48B | 72B |
| 10×10 (100) | 100B | 200B | 300B |
| 15×15 (225) | 225B | 450B | 675B |
| 20×20 (400) | 400B | 800B | 1200B |

→ **10×10・2〜3ノード構成が 250〜600 帯のスイートスポット**。これが実測で確定。

## 設計ルール（実測校正後の確定版）

1. **出力テンソルは必ず uint8**（memory 1/8）。入力dtypeは自由（memory非計上）。
2. **色置換は int32入力 + Gather単発**（recolor_direct）。Cast を足さない。
3. **論理合成は Equal/Xor でマスク → Where で適用**（Where は MAC=0）。
4. **ノード数＝中間出力数を減らす**（融合で中間テンソルのmemoryを削る）。
5. **定数（Slice/Gather）の params も計上される**ので、不要な軸指定を省く。

## 注意：本番採点器との差分確認

本校正は手元の onnx-tool で実施。コンペ本番の採点器が
- 同じバージョンか
- 複数デモペアをどう集計するか（1グラフを全ペアに適用 or バッチ）
- 入力テンソル非計上が本番でも同じか
は、提出して実スコアと照合して最終確認すること。
