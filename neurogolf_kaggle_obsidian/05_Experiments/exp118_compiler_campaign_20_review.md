# exp118_compiler_campaign_20_review

## Purpose

compiler campaign #20の必須local estimate測定と、#16〜#20のレビュー。

## Result

- local delta #1〜#20: `+0.1239388580395987`
- local delta #16〜#20: `0.000000`
- current submit-safe working local: `6282.93615685804`

## Verdict

`strategically_useful_but_no_local_gain`

#16〜#20はlocalを増やしていない。しかしtask185について、続行/撤退判断に必要な事実を得た。

- core lowering: cost `1516`
- fixed coordinate: fail
- Python window selector: `267/267`
- direct ONNX selector: too expensive (`4204` / `5160` / `76194`)

## Next Policy

#21〜#25はtask366/task365へpivotする。task185はfused selectorの新案が出た場合のみ戻る。

## Risk

- leakage risk: low。
- overfitting risk: medium。raw position table化を避ける。
