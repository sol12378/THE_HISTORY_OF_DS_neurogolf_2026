# exp327 task037 bounded diagonal shift lowering

## 目的

task037 `opposite_ray_diag_same_color` rule を、過去の full-grid Conv visibility ではなく bounded diagonal shift + equality mask で lowering し、current baseline より低costになるか確認する。

## 仮説

距離1〜5の diagonal endpoint を Slice/Pad shift で参照し、両側 endpoint が同じ one-hot 色かつ間が0なら埋める構成なら、Conv visibility cost `441976` より下がる可能性がある。

## 結果

- base: exp297 current task037
- base cost: `63726`
- candidate: `bounded_diag_shift_d5`
- validation: `266_pass_0_fail`
- candidate cost: `6633317`
- candidate points: `9.292384461287979`
- local delta: `0.0`
- submission: no submit

## 判断

functionally correct だが cost は完全に不採用。Slice/Pad shift と Mul 条件を大量に積むと full-grid intermediate が増え、GridSample/shift の狙いとは逆に memory cost が爆発する。

task037 は rule としては良いが、dense shift-stack lowering は停止する。次に task037 を続けるなら、changed-cell sparse construction、per-diagonal compact table、または current artifact 由来の低cost graph surgery が必要。compact form が見えなければ task185/048 へ pivot する。

## Risk

- leakage risk: low。入力のみの説明可能 rule。
- overfitting risk: medium。full-arc pass だが task-specific であり、改善時は Kaggle calibration が必要。

## Next

task037 dense shift-stack は止め、compact diagonal representation を設計するか、GridSample queue の task185/048 へ pivot する。
