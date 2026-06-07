# exp017_program_synthesis_seed_6500 notes

## 仮説

既存の1-step neighbor fillでは拾えない線延長・穴埋め系taskを、ONNXへ静的compile可能なiterative neighbor-fill DSL primitiveで拾える可能性がある。

## 判定

- baseline local estimate: `6479.3830863527055`
- new local estimate: `6479.3830863527055`
- gap to 6500: `20.616913647294496`
- improved tasks: `[]`

## Risk

- baseがexp016のためleakage/overfitting riskは高い。
- 6500到達時もsubmit前にfull arc-gen validationとstrict-risk監査が必要。
