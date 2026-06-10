# exp132_farm_step1_ir_emit_score

## Hypothesis

実taskに対してIRProgram -> ONNX emitter -> official score/evaluate_candidateを接続すれば、farm OSの最初の実パイプラインとして進捗確認できる。

## Result

- status: `step1_complete`
- n_programs: `7`
- n_improved: `0`

- task001 identity: status `rejected`, validation `0_pass_1_fail`, cost ``, proxy `0`
- task002 identity: status `rejected`, validation `0_pass_1_fail`, cost ``, proxy `0`
- task003 identity: status `rejected`, validation `0_pass_1_fail`, cost ``, proxy `0`
- task004 identity: status `rejected`, validation `0_pass_1_fail`, cost ``, proxy `0`
- task005 identity: status `rejected`, validation `0_pass_1_fail`, cost ``, proxy `0`
- task001 channel_gather: status `rejected`, validation `0_pass_1_fail`, cost ``, proxy `10`
- task001 static_slice_pad: status `rejected`, validation `0_pass_1_fail`, cost ``, proxy `381`

## Interpretation

Step 1は接続性確認であり、score-producing候補の採用はまだ行わない。identity/channel_gather/static_slice_padが実taskの既存評価関数へ流れ、validationとofficial score結果がCSV化されれば成功。

## Next

Step 2でfull-arc validationを明示接続し、BundleCandidateへ変換する。
