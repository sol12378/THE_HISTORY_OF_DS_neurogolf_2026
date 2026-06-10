# exp297 exp262 skip task048/336 fresh candidates

## 目的

exp262 rank221-320 sweep は task048 周辺で停止していた。runtime-stopper task048 と task336 を skip して完走し、exp263 partial で提出済みの17件を除いた fresh candidate があるか確認する。

## 結果

- 元実行: `experiments/e297_short_skip048_336/`
- fresh bundle: `experiments/exp297_exp262_skip_task048_336_fresh_candidates/`
- fresh selected tasks: `60, 226, 123`
- local delta: `+0.05520869941921447`
- expected public LB if calibrated: `6008.955208699419`
- full-arc replay: task060 `23_pass_0_fail`, task226/task123 `24_pass_0_fail`
- zip sanity: 400 files / names OK / sha256 `f08828ec01b436a87c72eeafc95b001c84c538771de7ed3d26312c7962a100ae`

## 判断

現行 best `exp265` public LB `6008.90` を local estimate で上回るため提出対象。Kaggle ref `53535771` として提出し、public LB `6008.96` で COMPLETE。期待 `6008.955208699419` と一致し、current best を更新。

## リスク

- leakage risk: low-medium。既存 artifact に対する full-arc gated graph surgery replay。
- overfitting risk: medium-low。公開LB較正は過去の bypass bundle とほぼ一致しているが、微小deltaなので表示丸めの影響はある。
