# exp_b023_seddik_surgery_pattern_audit

## 目的

`seddiktrk/surgical-onnx-precision-parameter-reduction` を、strict seedへ使えるgraph surgery/precision削減patternの知識源として読む。

## 抽出pattern

- `initializer_pruning`: count=53; action=strict seed全taskでunused initializer / duplicate initializer / scalar initializer化を再監査する。; risk=low-to-medium: validationとstatic check必須。
- `compress_or_banned_op_awareness`: count=15; action=candidate emission前にbanned op検出を強化し、Compress系artifactをsubmit candidateから外す。; risk=low: guardrail強化。
- `precision_parameter_reduction`: count=56; action=strict seedからinitializer dtype/constant size削減候補を生成し、full-arc gateで評価する。; risk=medium: dtype変更は数値/argmax挙動を壊す可能性。
- `score_cost_loop`: count=45; action=exp_bのcandidate evaluationにprofile/cost failure reasonをより細かく記録する。; risk=low.
- `pad_conv_where_patterns`: count=3; action=massimiliano/vyanktesh notebookと合わせてcheap Slice/Gather/Pad lowering patternを抽出する。; risk=medium: naive full-grid patternsはb011のように高cost化。

## Decision

まずstrict seed全taskに対して、unused/duplicate/scalar initializerなど低リスクなparameter surgery auditを行う。dtype変更はfull-arc gate付きの後段に回す。
