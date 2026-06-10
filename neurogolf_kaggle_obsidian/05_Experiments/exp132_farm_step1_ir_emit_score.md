# exp132_farm_step1_ir_emit_score

## Hypothesis

実taskに対して `IRProgram -> ONNX emitter -> official score/evaluate_candidate` を接続すれば、farm OSの最初の実パイプラインとして進捗確認できる。

## Result

- Step: 1 / 3
- Scope: IRProgram -> ONNX emitter -> `evaluate_candidate`
- arc-gen sample: `1`
- programs: `7`
- improved: `0`

実行した候補:

| task | family | validation | status | proxy |
|---|---|---|---|---:|
| task001 | identity | 0_pass_1_fail | rejected | 0 |
| task002 | identity | 0_pass_1_fail | rejected | 0 |
| task003 | identity | 0_pass_1_fail | rejected | 0 |
| task004 | identity | 0_pass_1_fail | rejected | 0 |
| task005 | identity | 0_pass_1_fail | rejected | 0 |
| task001 | channel_gather | 0_pass_1_fail | rejected | 10 |
| task001 | static_slice_pad | 0_pass_1_fail | rejected | 381 |

## Interpretation

接続性確認は成功。IRからONNXをemitし、既存の公式互換評価関数へ流し、validation/score結果をCSV化できた。一方、候補はすべて正解性で落ちたので採用候補ではない。これは想定内で、Step 1の目的は「実taskに直結する配管の確認」である。

## Next

Step 2では、解ける可能性がある候補だけをfull-arc validationへ流し、`BundleCandidate`へ変換する。identityのような明らかなcontrolはledgerに入れない。

## Leakage / Overfitting Risk

採用候補なし。local estimate/LBへ加算しない。
