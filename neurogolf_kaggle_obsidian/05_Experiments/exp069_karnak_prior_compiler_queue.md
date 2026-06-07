# exp069_karnak_prior_compiler_queue

## 目的

Karnak task description libraryを `exp053` cost target queue、`exp054` signature taxonomy、`exp068` current manifestへjoinし、250〜600化に効くcompiler lane優先順位を作る。

## 結果

- queue tasks: 400
- top lane by `gain_to_250`: `LOCAL_PREDICATE_FILL_COMPILER`
- LOCAL_PREDICATE_FILL: 139 tasks, gain_to_250 `616.93`
- CROP_SHAPE: 141 tasks, gain_to_250 `510.15`
- OBJECT_MOVE: 53 tasks, gain_to_250 `198.11`

## Top LOCAL_PREDICATE_FILL Tasks

`173, 77, 158, 133, 71, 285, 85, 251, 383, 54`

## Decision

次はLOCAL_PREDICATE_FILL上位taskのchanged-cell feature profileを作る。`exp066` task020型のbbox-local/color-role fill compilerが横展開できるかを確認する。
