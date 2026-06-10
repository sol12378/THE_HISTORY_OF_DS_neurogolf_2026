# exp266_dtype_memory_probe

## 目的

同一構造の中間tensor dtypeだけを変え、official costがdtype比例で下がるか確認する。

## 結果

- cast_fp32: memory `None`, params `None`, cost `None`, reason `score returned none`
- cast_fp16: memory `None`, params `None`, cost `None`, reason `score returned none`
- cast_int64: memory `None`, params `None`, cost `None`, reason `score returned none`
- cast_uint8: memory `None`, params `None`, cost `None`, reason `score returned none`
- bool_notnot: memory `None`, params `None`, cost `None`, reason `score returned none`
- arith_fp32: memory `None`, params `None`, cost `None`, reason `score returned none`
- arith_fp16: memory `None`, params `None`, cost `None`, reason `score returned none`
- arith_int64: memory `None`, params `None`, cost `None`, reason `score returned none`
- arith_uint8: memory `None`, params `None`, cost `None`, reason `score returned none`

## 判断

このprobe結果をもとに、dtype縮小post-passのgo/no-goを決める。
