# exp_b004_teacher_gain_p0_structural_taxonomy

## 目的

teacher-gain P0 18 taskについて、次に追加すべき説明可能rule grammarを構造から決める。

## 結果

- profiled tasks: 18
- grammar counts: {'sparse background fill with object-role/orbit grammar': 1, 'sparse color-role fill with local neighborhood predicates': 8, 'shape-transform/object crop grammar': 7, 'object movement/copy grammar': 1, 'general sparse/object completion grammar': 1}

## Top Priority

| task | gain | grammar | lowering | note |
|---:|---:|---|---|---|
| 20 | 3.132 | sparse background fill with object-role/orbit grammar | tiny ScatterND or one small static mask; avoid full-grid Where chains | highest: likely cheap if rule is found |
| 126 | 3.062 | sparse color-role fill with local neighborhood predicates | small Conv kernels or tiny coordinate program | high: changed cells are sparse but role inference is needed |
| 173 | 3.029 | sparse color-role fill with local neighborhood predicates | small Conv kernels or tiny coordinate program | high: changed cells are sparse but role inference is needed |
| 77 | 2.712 | shape-transform/object crop grammar | constant Slice/Gather when shape rule is static; reject full-grid bbox GatherND | medium: first infer shape rule |
| 366 | 2.627 | shape-transform/object crop grammar | constant Slice/Gather when shape rule is static; reject full-grid bbox GatherND | medium: first infer shape rule |
| 71 | 2.614 | object movement/copy grammar | small object masks and static placement; avoid large ScatterND | medium: object correspondence first |
| 206 | 2.550 | shape-transform/object crop grammar | constant Slice/Gather when shape rule is static; reject full-grid bbox GatherND | medium: first infer shape rule |
| 64 | 2.361 | shape-transform/object crop grammar | constant Slice/Gather when shape rule is static; reject full-grid bbox GatherND | medium: first infer shape rule |
| 370 | 2.238 | shape-transform/object crop grammar | constant Slice/Gather when shape rule is static; reject full-grid bbox GatherND | medium: first infer shape rule |
| 251 | 2.224 | sparse color-role fill with local neighborhood predicates | small Conv kernels or tiny coordinate program | high: changed cells are sparse but role inference is needed |

## 解釈

P0は単純なD4/rectangle/lineでは足りない。次は、変更セル数・bbox保存・背景fill条件を利用したobject-role/local-neighborhood grammarを追加する。
