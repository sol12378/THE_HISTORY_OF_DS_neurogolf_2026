# exp134_known_good_supplier_farm_acceptance

## Hypothesis

exp127のknown-good accepted候補をsupplierとしてfarmへ再投入すれば、full-arc validation/official score/BundleLedger acceptedまで通る。

## Result

- status: `known_good_supplier_complete`
- n_candidates: `2`
- n_full_ok: `2`
- n_accepted: `2`
- total_local_delta: `0.002854250468491415`

- task263 fourth_bypass_Where_114_in2: full `267_pass_0_fail`, official `8040`, accepted `True`
- task316 fourth_bypass_And_024_in0: full `266_pass_0_fail`, official `3833`, accepted `True`

## Interpretation

control候補ではなくknown-good候補を入れると、farm ledgerがacceptedを出せることを確認した。これで候補supplierさえ強ければ、pipelineは採用・micro submission候補化まで進められる。

## Leakage / Overfitting Risk

exp127由来の既知full-validation gated graph surgery候補。新規LB加算ではなくpipeline acceptance再現実験。
