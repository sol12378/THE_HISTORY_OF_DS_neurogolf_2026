# exp244_task391_static_template_audit

## 目的

task391は3x1固定出力・stable-binary候補なので、出力templateと色が入力の簡単な特徴で決まるか監査する。

## 結果

- baseline_cost: `10021`
- output_shape: `3x1`
- binary_signature_count: `1`
- binary template: `(1, 1, 1)`
- colorized_signature_count: `1`
- output_color_hist: `{1: 23, 2: 36, 3: 34, 4: 26, 5: 21, 6: 38, 7: 31, 8: 29, 9: 29}`
- best color rule: `bbox_bl 49/267`

## 判断

出力形状/templateは固定だが、色selectorが未解決。static 3x1 template loweringには進まない。

## リスク

色selectorを雑にtable化すると過学習になる。別の構造特徴が必要。
