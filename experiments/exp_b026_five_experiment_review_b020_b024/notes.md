# exp_b026_five_experiment_review_b020_b024

## 目的

5実験ごとのPDCA点検として、`exp_b020` から `exp_b024` が本当に `cost<=250` / 7700 / local-LB統一に有意義だったかを確認する。

## Reviewed

- `exp_b020_l4_shape_crop_pattern_profiler`
- `exp_b021_strict_seed_exp038_micro_delta_submit`
- `exp_b022_public_notebook_intelligence`
- `exp_b023_seddik_surgery_pattern_audit`
- `exp_b024_safe_uniform_initializer_scalarization`

## 判定

有意義。ただし7700本筋への寄与は「post-passと較正」が中心で、cost<=250を大量達成するrule/compiler本体ではまだない。

## 良かった点

- `exp_b020` はL4上位が単純固定crop/bbox cropではないことを確認し、安易なSlice路線を下げた。
- `exp_b021` はfull-arc-safe graph surgery micro deltaを提出し、LBで方向が維持されることを確認した。
- `exp_b022` はユーザー指定notebookを単なる再blendではなく、source reliability / lowering / taxonomy intelligenceへ分解した。
- `exp_b023` はSeddik notebookから具体的なinitializer surgery patternを抽出した。
- `exp_b024` はstrict seed上の狭いSeddik-style post-passを提出し、local `+0.027485` とLB `+0.03` が一致した。

## 足りない点

- score gainは小さい。
- cost<=250達成task数は増えていない。
- L4 / L2 rule compilerの本体探索はまだ停滞している。

## 次PDCA

- submit-safe delta unionは続けてbest bundleへ積む。
- Seddik-style post-passは標準後処理にする。
- 主戦略はobject-anchor crop、L3 object move/erase、task020型sparse object completionの横展開へ戻す。
- public notebook由来のConv/Slice/Gather loweringsは、既存低cost artifact profileと組み合わせて使う。
