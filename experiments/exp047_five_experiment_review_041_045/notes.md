# exp047_five_experiment_review_041_045

## 目的

ユーザー指定の「5実験ごとに本当に7700達成へ有意義だったか問い直す」ゲート。

## 対象

- exp041: task145 deeper surgery
- exp042: lightweight DSL/DAG proxy scan
- exp043: fixed DSL hit lowering
- exp044: task031 dynamic bbox full-grid lowering
- exp045: rule-searcher best practices / 7700 projection

## 判定

mixed but useful。

exp041の追加surgery自体は7700へほぼ効かないが、提出LB 3417.71により、exp016-041系local upperがsubmit-safeでないことを確定した点は極めて重要。exp042はDSL/DAG方向の可能性を示したが、exp043/044でproxyと実ONNX costの乖離が大きいことも確認した。exp045は7700をfamily-level cost targetへ再定義したため有用。

## 反省

- これ以上、lookup artifactのlocal scoreを磨く実験は主経路にしない。
- 「正しいrule」を見つけても、ONNX loweringが大きければ無価値。
- fixed simple transformは既存artifactが強く、主戦場ではない。

## 次の5実験方針

1. submit-safe seed inventoryを作る。
2. lookup由来taskはteacherとして使い、明示ruleへ圧縮する。
3. all arc-genを探索用と監査用に分けるfamily holdoutを作る。
4. object/crop compilerは既存artifact profileから小さいpatternを抽出する。
5. line/region/point patternを閉形式maskへ落とす。

## リスク

- leakage risk: review only
- overfitting risk: review only
