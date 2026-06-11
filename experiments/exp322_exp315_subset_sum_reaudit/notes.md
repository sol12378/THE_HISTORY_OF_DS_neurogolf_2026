# exp322_exp315_subset_sum_reaudit

## 目的

exp315 の missing drop を、exp318/319/321 の follow-up 結果込みで再監査する。

## 結果

- status: `completed`
- exp315 missing_drop_vs_all_alive: `27.760763322976885`
- exp318: task193/task275 は all-alive
- exp319: task101/task243 fail-stub は no-drop
- exp321: task101/task243 repair は no-gain

## Top Subsets After Exclusions

- tasks `[97, 137]` / sum `27.87930026163878` / error `0.11853693866189374`
- tasks `[137, 192]` / sum `27.950661017129526` / error `0.18989769415264135`
- tasks `[137, 197]` / sum `27.97559996547678` / error `0.21483664249989332`
- tasks `[137, 213]` / sum `27.98477625096774` / error `0.22401292799085581`
- tasks `[137, 324]` / sum `28.012125450386677` / error `0.25136212740979147`
- tasks `[137, 138]` / sum `28.020406007681988` / error `0.25964268470510277`
- tasks `[137, 335]` / sum `28.067249319565676` / error `0.3064859965887905`
- tasks `[137, 224]` / sum `28.08450033893277` / error `0.323737015955885`
- tasks `[137, 338]` / sum `28.094271179537763` / error `0.3335078565608782`
- tasks `[131, 137]` / sum `28.10467700775192` / error `0.3439136847750355`

## 判断

Do not spend another repair attempt on task101/task243 from public-code sources. The pair explains the no-drop probe arithmetically, but exp321 showed no recoverable public gain with the best full-arc source. Treat exp315 as diagnostic but not yet a reliable repair queue; resume fresh wide probing or move to GridSample queue.

## 次

exp323 fresh wide bisection probe on a new unprobed high-risk group, with pair-wise subset-sum uniqueness checked before submission; alternatively start task251 GridSample if submission budget or probe confidence is constrained.

## リスク

- leakage risk: low: analysis uses only prior public submission scores and local experiment metadata.
- overfitting risk: medium: the audit reasons about public LB diagnostics, so it should guide probe design rather than direct final model selection.
