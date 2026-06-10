# exp227_task300_exp226_output_diagnostic

## 目的

exp226のtask300 ONNX候補がexample 0でmismatchする原因を、出力gridの位置/色から診断する。

## 結果

- argmax_equal_padded: `True`
- expected_top_left: `[[2, 2, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [0, 2, 2, 0, 0, 0], [2, 2, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]`
- pred_top_left_argmax: `[[2, 2, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [0, 2, 2, 0, 0, 0], [2, 2, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]`

## 判断

result.jsonのtop-left比較を見て、色選択・bbox index・出力decodeのどこがずれているかを決める。
