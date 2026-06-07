# exp013_node_profile_and_rewrite_targets

## 目的

exp012後のtop cost taskをONNX node単位でprofileし、次にtemplate化・graph surgeryすべきtaskとop patternを決める。

## 結果

- top100をprofileした。
- `node_profile.csv`, `rewrite_target_manifest.csv`, `result.json`, `notes.md` を保存した。
- node rows: `5966`
- route内訳: sparse/object `65`, crop/resize `29`, same-shape `6`

## 判断

top costの主戦場は sparse/object completion と crop/resize。same-shapeは数は少ないが、task187/task198のようにcostが大きく、6500到達には個別対応が必要。

## Risk

profile自体は低risk。ただし以降のrewriteはexp012のsample-local upper boundをbaseにするため、leakage/overfitting riskは高い。
