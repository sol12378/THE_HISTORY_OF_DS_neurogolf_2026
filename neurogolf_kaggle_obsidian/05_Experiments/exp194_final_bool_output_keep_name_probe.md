# exp194_final_bool_output_keep_name_probe

## 結果

- task206: validation `266_pass_0_fail`, cost `61386 -> 61386`
- task328: validation `267_pass_0_fail`, cost `68053 -> 68053`
- accepted_count: `0`

## 判断

最終output Cast除去だけではofficial costが下がらない。Phase C dtype圧縮は内部full-grid中間またはfresh loweringを対象にする。
