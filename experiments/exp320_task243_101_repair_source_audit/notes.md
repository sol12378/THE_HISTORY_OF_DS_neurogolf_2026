# exp320_task243_101_repair_source_audit

## 目的

exp319 で task243/task101 を fail-stub しても Public LB が `6008.96` から落ちなかったため、この2 task を public-zero repair target として既存 accepted candidate source を監査する。

## 結果

- status: `completed`
- base_public_lb: `6008.96`
- exp319 public LB: `6008.96`
- interpretation: task243/task101 は current-best lineage 上で public-zero と扱う
- expected_public_gain_if_both_fixed: `27.779018459001815`
- expected_public_lb_if_both_fixed: `6036.739018459002`

## Top Candidates

### task101
- accepted rows: `24`
- `franksunp_blended_best` / `franksunp/neurogolf-blended-best-models` / local_points `13.904485714156337` / cost `65875` / sha `1097ccf2575df8ef91fdfbd7870a10cc2b4bdce67da9e625e16e66f5aeb89204`
- `kojimar_5800_minimal_blend` / `kojimar/neurogolf-5800-55-minimal-onnx-blend-assets` / local_points `13.904485714156337` / cost `65875` / sha `1097ccf2575df8ef91fdfbd7870a10cc2b4bdce67da9e625e16e66f5aeb89204`
- `konbu17_blend_source_v360` / `konbu17/neurogolf-2026-blend-source-v3-6-0` / local_points `13.904424994937168` / cost `65879` / sha `1483f4a4cd1a7882571597cf8d368fd4f28c34e09f2745e85adfb38987775dde`
- `massimilianoghiotto_6254` / `massimilianoghiotto/neurogolf2026-6254` / local_points `12.166837180696787` / cost `374431` / sha `edd7e165314a501462e02577d29022d3eb27078c4eb090009d5f19a5f0c2fbf1`
- `vyanktesh_multi_source_output` / `vyankteshdwivedi/neurogolf-multi-source-onnx-solver` / local_points `12.166837180696787` / cost `374431` / sha `edd7e165314a501462e02577d29022d3eb27078c4eb090009d5f19a5f0c2fbf1`

### task243
- accepted rows: `28`
- `franksunp_blended_best` / `franksunp/neurogolf-blended-best-models` / local_points `13.874532744845478` / cost `67878` / sha `d82711ed6d973d2c76345cf69ab278ecb1b2999b9c2410e8227df32d34e4dbf7`
- `konbu17_blend_source_v360` / `konbu17/neurogolf-2026-blend-source-v3-6-0` / local_points `13.874532744845478` / cost `67878` / sha `d82711ed6d973d2c76345cf69ab278ecb1b2999b9c2410e8227df32d34e4dbf7`
- `kojimar_5800_minimal_blend` / `kojimar/neurogolf-5800-55-minimal-onnx-blend-assets` / local_points `13.874532744845478` / cost `67878` / sha `d82711ed6d973d2c76345cf69ab278ecb1b2999b9c2410e8227df32d34e4dbf7`
- `franksunp_super_blend_v2_output` / `franksunp/neurogolf-super-blend-v2` / local_points `13.874532744845478` / cost `67878` / sha `d82711ed6d973d2c76345cf69ab278ecb1b2999b9c2410e8227df32d34e4dbf7`
- `massimilianoghiotto_6254` / `massimilianoghiotto/neurogolf2026-6254` / local_points `13.401199636921024` / cost `108967` / sha `8c58cc318fa9f59c3692654d50085f7b92b0511a745028cf7bca8bfc694b9ca1`

## 判断

提出はしない。次実験で recommended_repair の top candidate を exp297 に差し替え、full validation と zip sanity の後、単体または2 task bundle repair として提出する。

## リスク

- leakage risk: medium-high。public-code/raw blend source 由来であり、private robustness は低めに見る。
- overfitting risk: medium。public diagnostic で特定した repair target なので、採用判断は full local validation と expected LB 一致に限定する。
