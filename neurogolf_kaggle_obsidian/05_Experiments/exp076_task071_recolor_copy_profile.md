# exp076_task071_recolor_copy_profile

## 目的

`task071` のmulti-color recolor/copyを、小さいONNXへ落とせるbranch tree候補へ分解する。

## 結果

- changed cells: `4260`
- best out-color feature: `dr+dc+h+w+in_color+nbr4`
- out-color majority purity: `0.9192`
- best zero-mask feature: `dr+dc+h+w+in_color+nbr4`
- zero-mask purity: `0.9399`
- keys: `3194`

## 判断

局所featureは強いが、`3194` keys はそのままだとlookup-likeで採用不可。zero maskとsource-color copy directionを分離し、branch圧縮した後でのみONNX loweringへ進む。

## Risk

- leakage risk: low。全arc-genを診断に使ったが、raw key lookupを提出物にしていない。
- overfitting risk: medium。高純度featureでもkey数が大きく、branch圧縮なしではhidden汎化が弱い。

## Next

zero/nonzero mask compilerとsource-color copy direction compilerを分けて再探索する。
