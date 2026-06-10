# exp169_next4m_failure_bisection_probe

## 目的

exp166 current bestをベースに、残りsubset頻度候補の `133/396/158/047` をfail-stub化してpublic-zero有無を測る。

## 結果

- status: `probe_zip_ready`
- targets: `[133, 396, 158, 47]`
- expected_drop_if_all_alive: `55.82112499003305`
- expected_lb_if_all_alive: `5912.358875009967`
- zip sha256: `addf06b769617d0e6c6f66dd7e0e5c624967a33945ac71066caac5dee70a5a9b`
- kaggle_ref: `53523802`
- public_lb: `5938.07`
- observed_drop: `30.110000000000582`
- missing_drop_vs_all_alive: `25.71112499003247`
- interpretation: `task133 + task158` の合計点 `25.708777223030214` と一致するため、`task133/task158` はpublic-zero疑い、`task396/task047` はpublic-scoring alive。

## Target Validation

- task133: `0_pass_1_fail`, reason `mismatch example 0`
- task396: `0_pass_1_fail`, reason `mismatch example 0`
- task158: `0_pass_1_fail`, reason `mismatch example 0`
- task047: `0_pass_1_fail`, reason `mismatch example 0`

## 判断

task133/task158のrepairへ進む。task396/task047は immediate public-zero suspect から除外する。

## リスク

low: deliberate fail-stub probe; no private labels used.
medium: consumes a submission and only measures public scoring behavior for a group.
