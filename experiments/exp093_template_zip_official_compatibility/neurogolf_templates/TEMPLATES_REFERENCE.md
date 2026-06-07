# NeuroGolf 2026 — uint8最小テンソル ONNXテンプレート集 リファレンス

解釈A（1タスクあたり cost 250〜600）を狙う、タスク類型別のONNXグラフ生成テンプレート。
全テンプレートは `templates.py` に実装、`test_templates.py` で15/15が正しさ検証済み。

## 設計の3原則

1. **FREE opのみ** — Transpose / Slice / Gather / Tile / Concat / Pad。params≈0・MAC=0。よって cost ≒ memory（バイト総和）。
2. **uint8で統一** — 色値0–9はuint8（1バイト）で十分。int64比で8分の1。これがcost帯到達の最大要因。
3. **中間テンソル最小化** — memory項は全テンソル（入力＋中間＋出力＋定数）のバイト総和。ノード数を削るほど安い。

## 確定したコスト計算式

```
cost ≒ memory(bytes)
     = Σ(各テンソルの要素数 × dtypeバイト数)
     = 入力テンソル + 各ノード出力 + 定数(initializer)
```

10×10グリッド・uint8なら 1テンソル=100バイト。Sliceの定数（starts/ends/axes/steps）はint64で各8バイト程度の小口。

## テンプレート別 cost予算表（10×10グリッド・uint8で実測概算）

| # | テンプレート | op構成 | ノード | cost(10×10) | score | 帯 |
|---|---|---|---|---|---|---|
| T1 | identity | Identity | 1 | 約200 | 19.70 | <250 |
| T2 | flip_lr | Slice(-1) | 1 | 約232 | 19.55 | <250 |
| T3 | flip_ud | Slice(-1) | 1 | 約232 | 19.55 | <250 |
| T4 | rot180 | Slice(-1,-1) | 1 | 約264 | 19.42 | ★250-600 |
| T5 | rot90_ccw | Transpose+Slice | 2 | 約332 | 19.19 | ★250-600 |
| T6 | rot90_cw | Transpose+Slice | 2 | 約332 | 19.19 | ★250-600 |
| T7 | transpose | Transpose | 1 | 約200 | 19.70 | <250 |
| T8 | recolor | Gather+LUT | 1 | 約510 | 18.77 | ★250-600 |
| T9 | tile | Tile | 1 | 約216＋ | 19.62 | <250（出力拡大で増） |
| T10 | crop | Slice | 1 | 約248 | 19.49 | <250 |
| T11 | pad | Pad | 1 | 約233 | 19.55 | <250 |
| T12 | concat_self | Concat | 1 | 約200＋ | 19.70 | 出力拡大で増 |
| T13 | mirror_concat | Slice+Concat | 2 | 約332＋ | 19.19 | ★250-600 |
| T14 | downscale | Slice(step=k) | 1 | 約200 | 19.70 | <250 |
| T15 | upscale | Gather×2 | 2 | 約332＋ | 19.19 | 出力拡大で増 |

## 重要な発見

**多くの単純タスクは cost 250 を下回る**（identity, flip, transpose, crop, pad など）。これは「削りすぎ」ではなく**満点に近く理想的**。250〜600帯は「これ以上削れない複雑タスクの目標」であって、単純タスクが250未満なのは歓迎すべき状態。

**色置換(recolor)が510と帯の上の方**なのは、入力をint32（グリッド値がGatherのindexになるため整数型必須）で受けるため入力テンソルが10×10×4=400バイトになるから。出力はuint8。これは型制約上やむを得ない。

**出力が拡大するタスク**（tile, concat, upscale）は出力テンソルのバイトが入力より大きいため、サイズ次第で帯に入る。2×2タイルなら出力は4倍。

## グリッドサイズによる帯の移動

同じテンプレートでもグリッドが大きいとcostが上がる（uint8・2テンソル構成の場合）:

| サイズ | 1テンソル | 2テンソル | 3テンソル |
|---|---|---|---|
| 5×5 | 25B | 50B | 75B |
| 10×10 | 100B | 200B | 300B |
| 15×15 | 225B | 450B | 675B |
| 20×20 | 400B | 800B | 1,200B |
| 30×30 | 900B | 1,800B | 2,700B |

→ 15×15以上のタスクは2〜3ノードで帯を超える。ノード数を1に絞るか、超過を容認する（cost1,200でもscore17.9で、帯との差は0.7点）。

## 使い方

```python
import templates as T
import numpy as np

# 例: あるタスクが「左右反転」だと判明したら
model = T.t_flip_lr()

# ローカルで正しさ検証（全trainペアで）
for x, y_true in train_pairs:
    y = T.run(model, x.astype(np.uint8))
    assert np.array_equal(y, y_true)

# テンソル数を確認（cost概算の根拠）
ntensor, nnode, ninit = T.count_tensors(model)

# 保存して提出
import onnx
onnx.save(model, "task000.onnx")
```

## 次の拡張（このテンプレート集の外）

- **複合変換**: 上記の合成（例 flip→recolor）。中間テンソルが増える分costも増えるので、ノード融合で抑える。
- **論理系（Tier B）**: XOR/AND/OR のグリッド合成は CHEAP op（出力要素数ぶんのMAC）。本テンプレート集はFREE opのみなので別系統。
- **動的形状の色置換**: recolorのint32入力問題を回避するため、グリッド自体をuint8で持ちつつ別経路でindex化する方法は要検討。
- **onnx-tool実測校正**: 本表のcostは「テンソル要素数×dtype」の概算。実際のonnx-toolが入力テンソルや定数をどう計上するか、§probe で実測して絶対値を校正すること。
