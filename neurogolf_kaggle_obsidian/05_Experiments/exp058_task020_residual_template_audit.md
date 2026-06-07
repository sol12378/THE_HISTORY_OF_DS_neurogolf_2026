# exp058_task020_residual_template_audit

## Hypothesis

task020のD4 orbit ruleが漏らす例は、少数のbbox-local templateに分解できる。

## Result

- examples: `266`
- unique D4-canonical templates: `3`

| Rank | Count | D4 pass | D4 fail | Template |
|---:|---:|---:|---:|---|
| 1 | 92 | 56 | 36 | `((0, 2), (2, 0), (2, 4))` |
| 2 | 87 | 49 | 38 | `((0, 0), (0, 4), (4, 0))` |
| 3 | 87 | 55 | 32 | `((1, 1), (1, 3), (3, 1))` |

## Interpretation

task020は、3-cell fill template自体は3種類に限定される。次の課題は、入力側の色配置・object role・local frameから、この3 templateのどれを使うか選ぶselectorを作ること。

template tableをそのままONNXに出すとmemorization riskが高い。rule化できるselectorだけを採用する。

## Decision

次は3-template selector探索。full passしたらtiny mask/ScatterNDでsingle-task deltaを作り、Kaggle較正提出する。
