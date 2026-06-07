# exp048_submit_safe_seed_inventory

## 仮説

exp041のLB崩壊後は、exp005 strict full-validation bundleをsubmit-safe seedとし、exp023/041系はteacherとしてのみ使うべき。

## 結果

- strict seed score: 6282.230228
- teacher local score: 6479.398825
- teacher gain vs strict: 197.168597
- P0 teacher-gain tasks: 18
- P1 high strict-cost tasks: 128

## 解釈

exp005は400/400 all arc-gen pass済みの信頼seed。exp023/041はlocal scoreが高いがLB崩壊済みなので、その差分は「答え」ではなく「圧縮すべきteacher signal」として扱う。

## 次

P0 taskをfamily別にrule-compressする。最初はteacher gainが大きく、routeが `sparse_edit_or_object_completion` または `crop_or_resize` のtaskから始める。

## リスク

- leakage risk: strict seedは低、teacherは高。
- overfitting risk: family holdout未導入のため中。
