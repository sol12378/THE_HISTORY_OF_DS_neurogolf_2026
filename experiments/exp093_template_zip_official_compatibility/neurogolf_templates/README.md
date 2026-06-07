# NeuroGolf 2026 — ONNXテンプレート集（実測校正済み）

ARC-AGI-1 タスクを uint8固定・最小テンソル構成で解く ONNXグラフ生成テンプレート。
全テンプレートは本物の onnx-tool で実測検証済み（18/18 正常動作）。

## ファイル構成

| ファイル | 内容 |
|---|---|
| `templates.py` | 基本テンプレート（恒等/反転/回転/転置/色置換/タイル/切出/パディング/連結/スケール）15種 |
| `templates_composite.py` | 複合変換（FREE op合成・ノード融合）C1〜C5 |
| `templates_logical.py` | Tier B 論理系（XOR/AND/OR/マスク塗り、Where活用）L1〜L4 |
| `templates_recolor_dynamic.py` | 動的形状の色置換（int32入力Gather単発が最安）R1〜R3 |
| `test_templates.py` | 基本テンプレートの正しさ検証 |
| `test_all.py` | 全テンプレートの正しさ＋onnx-tool実測コスト測定 |
| `probe_onnx_tool.py` | op別の課金を onnx-tool で実測する校正スクリプト |
| `CALIBRATION_REPORT.md` | 実測校正の結果（確定版） |
| `TEMPLATES_REFERENCE.md` | 基本テンプレートのcost予算表 |

## クイックスタート

```bash
pip install onnx onnxruntime onnx-tool numpy

# 全テンプレートの正しさ＋実測コストを確認
python3 test_all.py

# op別の課金を実測（本番採点器との校正用）
python3 probe_onnx_tool.py
```

## 実測で確定した4大事実

1. **memory項は出力テンソルのバイトのみ**（入力は非計上）→ 色置換はint32入力が最安
2. **uint8で memory 1/8**（vs int64）→ 出力は必ずuint8
3. **FREE op（Transpose/Slice/Gather/Tile/Concat/Pad/Where）は MAC=0**
4. **Where だけ CHEAP系で唯一 MAC=0** → 論理合成の最終段はWhere

## 使い方の例

```python
import templates as T
import templates_recolor_dynamic as R
import numpy as np, onnx

# タスクが「左右反転」なら
model = T.t_flip_lr()
for x, y in train_pairs:
    assert np.array_equal(T.run(model, x.astype(np.uint8)), y)
onnx.save(model, "task000.onnx")

# タスクが「色置換」なら（int32入力が最安）
model = R.recolor_direct([0,6,4,8,4,5,6,7,8,9])  # 長さ10のLUT
```

## 注意

cost値は手元の onnx-tool での実測。本番採点器のバージョン・複数ペア集計方法・
入力非計上ルールは、提出して実スコアと照合し最終確認すること。
詳細は CALIBRATION_REPORT.md を参照。
