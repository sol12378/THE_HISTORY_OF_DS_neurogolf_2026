# exp017_program_synthesis_seed_6500

## Purpose

6500まで残り20.62点の状態から、ONNXへ静的にloweringできる反復近傍fill primitiveで追加改善できるかを確認した。

## Result

- base: `experiments/exp016_top100_rewrite_campaign`
- top_k: `80`
- arc_gen_sample: `20`
- baseline local estimate: `6479.383086`
- new local estimate: `6479.383086`
- delta: `0.000000`
- improved tasks: `0`

## Interpretation

dry runのsample2では一部taskに小さな改善が出たが、sample20では採用ゼロだった。単純な近傍fillは残り高コストtaskの本質を捉えていない。

残り上位は、閉領域/外部領域の分離、線グリッドの部屋塗り、点から線やパターンを生成するタスクが多い。次はより構造的なDSL primitiveへ移る。

## Risk

- leakage risk: high。baseがexp016でsignature lookupを含む。
- overfitting risk: high。sample-localであり、full arc-gen前の評価。

