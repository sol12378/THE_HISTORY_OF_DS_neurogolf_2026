# exp062_task020_canon_pos_rule_compressor

## Hypothesis

task020の`canon_pos` 24 caseは、raw lookupではなく説明可能な小decision ruleへ圧縮できる。

## Result

- cases: `24`
- examples: `266`
- nonlookup full-pass hits: `1`
- rule:
  - if `corners_eq1` then `corners`
  - else if `edge_mid_eq1` then `edge_mid`
  - else if `has_inner` then `inner_diag`
  - else `corners`
- output pass: `266/266`

## Interpretation

これはtask020で初めてのsubmit候補に近い明示rule。raw 24-case tableではなく、orbit groupの数・存在だけでclassを決めるため、memorization riskはかなり下がった。

## Decision

次はtiny ONNX loweringを作り、official validationとcostを測る。cost 250〜600台に入り、strict seed task020より低costならsingle-task deltaとしてKaggle提出較正する。
