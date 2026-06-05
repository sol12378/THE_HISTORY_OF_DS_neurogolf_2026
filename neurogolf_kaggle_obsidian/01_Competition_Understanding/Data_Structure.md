# Data Structure

## Raw Inventory

- 2026-06-05: Kaggle APIから `data/external/neurogolf-2026/neurogolf-2026.zip` に取得し、`data/external/neurogolf-2026/unzipped` に展開。`data/raw` は変更していない。
- ファイルは `task001.json` ... `task400.json` と `neurogolf_utils/neurogolf_utils.py`。

## Expected Notes

- Task JSON schema
- Train/test example counts
- Grid dimensions
- Color/channel representation

## Task JSON Schema

- 各taskは `train`, `test`, `arc-gen` を持つ。
- 各subsetは `input`, `output` grid pairのlist。
- gridは1から30までの高さ/幅を持つ矩形listで、値は0から9の色index。
- network入力前に `[1, 10, 30, 30]` のfloat one-hot channel tensorへ変換される。
- 元gridの外側は全channel 0のzero-hotとして扱う。

## 初回観察

- `task087` は全公開例が3x3 input/outputで、変換は180度回転。
- 公開例数: train 4、test 1、arc-gen 261。
