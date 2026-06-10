# exp284_farm_current_best_public_lb_visibility

## 目的

farm の public-code registry 表示に、現行 Best Public LB `6008.90` を反映させる。

## 変更

- `PublicCodeRegistry.current_best_public_lb` を追加。
- `EXP_SUMMARY.md` の `Best Public LB` 行から `6008.90` を読む。
- `floor_status.max_observed_kaggle_lb` に `current_best_public_lb` を含める。
- `_float_or_none()` の切れた実装を修正。

## 結果

- `current_best_public_lb`: `6008.9`
- `max_observed_kaggle_lb`: `6008.9`
- `submit_floor_ready`: `false`
- `ready_source_count`: `0`

## 判断

成功。public-code source 自体は 6285 floor ready ではないが、farm の現状表示は現行 best LB と整合した。

## Submit

なし。tooling visibility のみ。smoke の `should_submit=true` は synthetic ledger のダミー判定であり、Kaggle candidate ではない。

## リスク

- Leakage risk: なし
- Overfitting risk: なし
