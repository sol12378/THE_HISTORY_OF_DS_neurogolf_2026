# exp324 task251 GridSample feasibility probe

## 目的

public-zero probe の直近収率が落ちたため、GridSample queue 先頭の task251 について、実ONNX `GridSample` が official scorer で安く使えるかを score-direct に確認する。

## 仮説

task251 は closed zero-component rule で解けるが、既存 flood-fill lowering は高costで負ける。`GridSample` が実 scorer でも低costなら、mask生成と組み合わせる次の lowering へ進める。

## 結果

- Python rule: `closed_zero_component_neighbor2_to_1`
- validation: `266_pass_0_fail`
- base cost: `100580`
- identity GridSample opset16/opset20: static ok / sanitize ok / official score ok
- GridSample identity cost: `1800`
- if correct near cost1800: expected delta `+4.023166765841701`
- GridSample identity validation: `0_pass_1_fail`
- task251 inputs with color1: `0/266`

## 判断

`GridSample` は実ONNXでも使える。proxy cost `9` ではなく、30x30 identity grid initializer込みの実測は `1800` だったが、それでも task251 baseline `100580` より十分安い。

ただし task251 は全例で入力に color `1` が無く、出力で closed zero component を color `1` に生成する問題なので、純粋な sampling だけでは解けない。次は closed-component mask 生成と cheap color1/recolor primitive の合成が必要。

## Submission

提出なし。identity GridSample は意図的に task251 不正解であり、feasibility probe のみ。

## Risk

- leakage risk: low。公式例での rule 再確認と primitive cost 測定のみ。
- overfitting risk: low-to-medium。task-specific rule は full-arc pass だが、この実験では置換候補を提出していない。

## Next

task251 を続けるなら `mask generation + cheap color1` の最小候補を作る。mask生成が高cost化するなら task037 GridSample target へ pivot する。
