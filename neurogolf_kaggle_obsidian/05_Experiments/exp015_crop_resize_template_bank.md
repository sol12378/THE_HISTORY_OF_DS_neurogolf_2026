# exp015_crop_resize_template_bank

## 目的

crop/resize task向けtemplate bankを実装し、task366, 396, 205, 216, 29, 109, 239, 201などを優先して圧縮する。

## 実装

- common fixed crop
- common fixed crop + color map
- global transform + color map
- signature ScatterND lookup
- constant sparse output

## 結果

- target task: `29`
- improved task: `25`
- accepted gainは主に `signature_scatternd_lookup`

## 判断

fixed crop系はstaticで安全だが、今回の対象では既存baseやlookupより低costになりにくかった。task398/task107/task233/task221などのremaining high cost crop taskは、固定cropではなくobject-aware cropやshape-dependent cropが必要。

## Risk

exp012 baseとsignature lookupに依存するため、local estimateは高risk。submit前にはfull validationとrule auditが必要。
