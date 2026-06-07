# exp_b025_submit_safe_delta_union

## 目的

Kaggle転移が確認済みのsubmit-safe deltaを合成して、現在の実運用best bundleを作る。

- base: `exp068_seddik_style_strict_scalarization`
- overlay: `exp_b021_strict_seed_exp038_micro_delta_submit`

## 結果

- accepted tasks: `62, 145, 255, 268`
- validation: all ok
- local delta over exp068: `+0.2018673343897568`
- new local estimate: `6282.812217709089`
- submission ref: `53418067`
- public LB: `5930.40`

## 解釈

`exp068` は task020 explicit rule + Seddik-style post-pass、`exp_b021` は exp038由来の4 task graph surgery deltaで、対象taskが衝突しない。最終zip上でも4 taskを再validationし、すべてfull-arc passした。

この合成は7700へ直接大きく進むものではないが、local/LB較正とsubmit-safe best更新には有意義。LBは `exp068` の `5930.27` から `+0.13` で、local delta `+0.201867` よりやや小さいが方向は維持された。明示rule/compiler本体で大きなcost削減を作った後も、このようなpost-pass/graph-surgery deltaを標準で重ねるべき。

## Risk

- Leakage risk: low-to-medium。2つの既提出full-arc-safe strict-derived deltaのunion。
- Overfitting risk: medium。task-specific graph surgeryなので、union時もLB較正が必要。

## Decision

Kaggle ref `53418067` はLB `5930.40` で完了。現時点のsubmit-safe bestとして扱う。
