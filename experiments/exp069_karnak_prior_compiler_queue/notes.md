# exp069_karnak_prior_compiler_queue

## 目的

Karnak task description libraryを、exp053 cost target queue・exp054 signature lane・exp068 current costへjoinし、cost 250〜600化のcompiler優先順位を作る。

## 結果

- queue tasks: 400
- top lane by gain_to_250: LOCAL_PREDICATE_FILL_COMPILER
- recommended next: `exp070_local_predicate_fill_compiler_priority_batch`

## 解釈

Karnakのdescriptionは直接の正解ではなく、探索順序を決めるpriorとして使う。`Object Detection`, `Color Mapping`, `Filling Regions` が多いので、`exp066` task020型のbbox-local/color-role fill compilerを横展開する価値が高い。

## Decision

次はLOCAL_PREDICATE_FILL_COMPILER上位taskに対し、changed-cell feature profileから小decision treeを合成する。full-arc exactかつcost改善したら、exp068 baseへ差し替えてSeddik post-pass後に提出する。
