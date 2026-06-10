# exp133_farm_step2_step3_validation_ledger

## Hypothesis

Step 2/3としてfull-arc validationとBundleLedgerを接続すれば、採用候補がない場合は安全停止し、採用候補がある場合だけ差分bundleを作れる。

## Result

- status: `pipeline_safe_operational`
- candidates: `7`
- full_ok: `0`
- accepted: `0`
- total_local_delta: `0`
- bundle_created: `False`

7候補はすべてsample validationで `0_pass_1_fail`。そのためfull-arc validationは `not_run`、ledgerは全件 `rejected`。accepted候補がないため `submission.zip` は生成しなかった。

## Interpretation

Step 2/3の配管は稼働した。重要なのは、候補が弱い場合にbundleを作らず安全停止できたこと。これにより、今後rule minerや既存rule-hit taskから候補を供給しても、full-arc validかつcost改善したものだけがbundleへ進む。

## Next

次はplausible rule candidateを供給する。候補源は以下が自然:

- exp066 task020 explicit rule系
- exp109/110/122のfocused surgery候補
- exp131で600級と確認した`static_slice_pad`を使う実task crop候補

## Leakage / Overfitting Risk

採用候補なし。local estimate/LBへ加算しない。
