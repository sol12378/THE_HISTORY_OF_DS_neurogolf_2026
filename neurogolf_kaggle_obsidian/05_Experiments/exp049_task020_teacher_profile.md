# exp049_task020_teacher_profile

## 目的

P0最大gainのtask020について、strict artifactとteacher artifactをprofileし、rule compressionの材料にする。

## 結果

- strict cost: 90,133
- teacher cost: 3,931
- teacher gain: +3.132393
- strict: 245 nodes, 25 initializers, 270 params
- teacher: 13 nodes, 6 initializers, 1031 params
- teacher ops: `GatherND`, `MatMul`, `ScatterND` を含むsignature lookup

## 解釈

teacherは安いが、signature lookupなのでsubmit-safeではない。task020の明示ruleを掘るためのoracleとしてのみ使う。

## 次

task020の入出力例を解析し、changed-cell mask / object completion ruleを探索する。
