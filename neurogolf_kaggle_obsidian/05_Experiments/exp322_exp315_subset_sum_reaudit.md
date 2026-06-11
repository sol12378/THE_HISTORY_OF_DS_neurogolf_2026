# exp322_exp315_subset_sum_reaudit

## 目的

exp315 の missing drop を、exp318/319/321 の follow-up 結果込みで再監査し、task101/task243 にさらに repair 時間を使うべきか判断する。

## 仮説

task101/task243 は exp319 の no-drop を算術的には説明するが、exp321 の repair no-gain により、recoverable public-zero としては扱いにくい。除外後の subset-sum が弱ければ、fresh wide probe に戻るべき。

## 結果

- status: `completed`
- exp315 missing_drop_vs_all_alive: `27.760763322976885`
- exp318: task193/task275 は all-alive
- exp319: task101/task243 fail-stub は no-drop
- exp321: task101/task243 repair は no-gain
- task193/task275 と task101/task243 を除外後の最小 error pair: task097+137、sum `27.87930026163878`、error `0.11853693866189374`

## 判断

task101/task243 への追加 public-code source repair は止める。この pair は exp315/exp319 の probe 算術を説明するが、exp321 で gain が出なかったため、修復可能な public-zero queue としては信頼しない。

exp315 は診断としては有用だが、これ以上の split/repair queue にはしない。次は fresh wide bisection probe で新しい未 probe 高リスク群を調べる。probe confidence や提出予算が気になる場合は task251 GridSample に切り替える。

## リスク

- leakage risk: low。既存 public score と実験 metadata の分析のみ。
- overfitting risk: medium。public LB 診断に基づくため、最終採用ではなく次 probe 設計に限定して使う。
