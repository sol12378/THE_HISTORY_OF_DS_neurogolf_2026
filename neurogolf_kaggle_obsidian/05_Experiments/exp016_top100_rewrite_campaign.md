# exp016_top100_rewrite_campaign

## 目的

exp014とexp015の候補を統合し、exp012 baseからlocal estimate `>= 6500` の `submission.zip` を作る。

## 結果

- base: `exp012_template_factory_core`
- top_k: `400`
- arc_gen_sample: `20`
- baseline local estimate: `6296.297369919829`
- new local estimate: `6479.3830863527055`
- delta: `+183.08571643287632`
- gap to 6500: `20.616913647294496`
- improved task: `195`
- zip sanity: 400 files, `task001.onnx` ... `task400.onnx` only

## 採用された改善

採用改善はすべて `signature_scatternd_lookup`。`global_transform_color_map`, `common_fixed_crop_color_map`, `constant_sparse_output` は実装・検証済みだが、今回のtop400では採用gainなし。

## Remaining Top Cost

- task286
- task187
- task198
- task203
- task398
- task313
- task255
- task107
- task137
- task29

## 判断

Phase 1の6500到達条件は未達。top400まで拡張しても追加gainは小さく、残り20.62点は高cost taskの個別graph surgeryなしには届きにくい。

## 次PDCA

- task187/task198/task137/task255向け: variable-shape same-shape transformをstatic ONNXで表現するtemplate。
- task286/task203/task313/task328向け: high-edit sparse/object completionをlookupではなくrule graphへ落とす。
- task398/task107/task233向け: object-aware cropをNonZero/Compressなしで近似する固定mask/bbox template。

## Risk

leakage risk: high。exp012/exp016はarc-gen sample labelを使ったlookup系改善を含む。

overfitting risk: high。local estimate最適化であり、Kaggle submit候補ではない。
