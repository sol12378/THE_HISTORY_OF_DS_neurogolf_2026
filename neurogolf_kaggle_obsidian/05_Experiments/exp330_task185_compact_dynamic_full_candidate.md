# exp330_task185_compact_dynamic_full_candidate

## Hypothesis

exp329 の bg-aware axis selector に、compact start/spacing dynamic index、`GatherElements` lattice extraction、homogeneous 2x2 core を接続すれば、task185 full candidate を baseline 未満で作れる。

## Result

| variant | validation | cost | status |
|---|---:|---:|---|
| `compact_dynamic_float` | `267_pass_0_fail` | `59784` | `no_cost_gain` |
| `compact_dynamic_uint8` | `267_pass_0_fail` | `95784` | `no_cost_gain` |

baseline cost は `59584`。best variant は baseline より `200` 高く、local delta は `-0.0033509849733857067`。

## Interpretation

correct full candidate は完成した。exp204 の巨大 `row_templates` / `col_templates` 依存は、`best_row/best_col -> start/spacing -> Tile` で解消できた。

ただし float candidate は baseline よりわずかに高いため提出しない。`uint8` final output は Cast / profiler memory が増えて逆効果だった。

## Decision

`no_submit`

次に task185 を続けるなら、あと `200` cost を削る narrow shave を行う。候補は以下:

- default-bg + Pad の memory を減らす。
- `Tile` で作る `row_idx` / `col_idx` の shape をさらに小さくする。
- selector score path の不要出力・中間を削る。

削れない場合は task251/task037 へ pivot する。

## Risk

- leakage risk: low。入力のみの bg detector、window selector、homogeneous local core。
- overfitting risk: medium。task-specific geometry だが full local examples で `267/267` pass。

## 10-Experiment Review Note

exp321-330 の LB 影響:

- exp321 repair は expected `6036.74` に対して LB `6008.96` で no-gain。
- exp323/325 の fresh public-zero probes は all-alive で、public-zero 高point順探索の収率低下を確認。
- exp327 は task037 correctness を証明したが cost wall。
- exp328/329/330 は task185 を detector -> selector -> full candidate まで進め、full validation pass まで到達したが cost gain はまだ `-0.00335`。

結論: 直近10実験の直接 LB gain は `0`。ただし exp330 により task185 は「未解決」から「あと 200 cost shave」に変わったため、次の1実験だけ task185 cost shave を許容し、失敗したら GridSample queue の別 task へ移る。
