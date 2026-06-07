# exp064_task020_rule_lowering_spec

## 目的

exp062のtask020明示ruleをONNX化する前に、dynamic bbox込みのlowering blockを明確にする。

## 結論

最初のONNX targetはcorrectness-first。acceptanceは `266/266` all arc-gen pass かつ strict task020 cost `90133` 未満。

cost 250〜600はstretch targetであり、初回はbbox/writebackが重くなる可能性が高い。

## Rule

- `corners_eq1 -> corners`
- `edge_mid_eq1 -> edge_mid`
- `has_inner -> inner_diag`
- default `corners`

## Next

1. exp046型のdynamic bbox cropで5x5 local frameを作る。
2. group countとclass selectorをONNX化する。
3. 3 template x D4 variant masksでmissing cellsを決める。
4. writebackして検証/cost計測する。
