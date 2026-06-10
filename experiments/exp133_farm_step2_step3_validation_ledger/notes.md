# exp133_farm_step2_step3_validation_ledger

## Hypothesis

Step 2/3としてfull-arc validationとBundleLedgerを接続すれば、採用候補がない場合は安全停止し、採用候補がある場合だけ差分bundleを作れる。

## Result

- status: `pipeline_safe_operational`
- n_candidates: `7`
- n_full_ok: `0`
- n_accepted: `0`
- total_local_delta: `0`
- bundle_created: `False`

- task001 identity: sample `0_pass_1_fail`, full `not_run`, adoption `rejected`
- task002 identity: sample `0_pass_1_fail`, full `not_run`, adoption `rejected`
- task003 identity: sample `0_pass_1_fail`, full `not_run`, adoption `rejected`
- task004 identity: sample `0_pass_1_fail`, full `not_run`, adoption `rejected`
- task005 identity: sample `0_pass_1_fail`, full `not_run`, adoption `rejected`
- task001 channel_gather: sample `0_pass_1_fail`, full `not_run`, adoption `rejected`
- task001 static_slice_pad: sample `0_pass_1_fail`, full `not_run`, adoption `rejected`

## Interpretation

Step 2/3の配管は稼働した。今回のcontrol候補は全てsample段階で不正解のため、full-arc validationへは進まず、ledgerも全件rejected。acceptedが0なのでsubmission.zipは生成しない。この安全停止が期待動作。

## Next

次はrule minerまたは既存rule-hit taskからplausible candidateを供給し、この同じpipelineでaccepted候補が出るか確認する。

## Leakage / Overfitting Risk

採用候補なし。local estimate/LBへ加算しない。
