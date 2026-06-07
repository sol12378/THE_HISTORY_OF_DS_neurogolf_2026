# exp079_task366_panel_overlay_probe

## 目的

`task366` を上下/左右2パネルoverlay ruleとして説明できるか調べる。

## 結果

- candidates: `22`
- best: `vertical_min` = `0/266`
- mean cell accuracy: `0.2850`

## 判断

full-pass候補があれば `Slice + Where` loweringへ進む。なければpanel overlayにmask/color-role branchを追加する。

## Risk

- leakage risk: low。手書きcandidateのみ。
- overfitting risk: medium。branch追加時はall-arc full pass必須。
