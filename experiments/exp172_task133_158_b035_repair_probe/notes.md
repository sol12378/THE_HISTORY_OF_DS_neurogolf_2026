# exp172_task133_158_b035_repair_probe

## 目的

exp169でpublic-zeroと推定された `task133` / `task158` を、exp_b035のdistinct full-local-valid rawで修復する。

## 結果

- status: `repair_zip_ready`
- expected_gain_if_public_pass: `25.638803`
- expected_lb_if_public_pass: `5993.818803`
- zip sha256: `3dd7b38f271740ad71b4c0807c37a8d83de219ec4e497cae0c5e43bb7b81becc`
- kaggle_ref: `53523884`
- public_lb: `5993.82`
- observed_gain_vs_base: `25.640000000000327`
- interpretation: 期待値と一致してrepair成功。exp166 `5968.18` から `+25.64`、current public bestを更新。

## Repair Validation

- task133: `267_pass_0_fail`, cost `196688`, sha `674c8a7eb1f001cc8db93a4b6a1b646561460bb557f887c2f7a6fbf37ecc1d44`
- task158: `266_pass_0_fail`, cost `193266`, sha `144ec47192d8fc076bdfeda2e2da8dfff38e3fca69435f3ce3221fbdaf7b2bf8`

## 判断

採用。current public bestをexp172へ更新する。高risk public/source blend repairなのでprivate robustness riskは継続して明記する。

## リスク

high: exp_b035 public/source blend candidate; public-zero repair may not imply private robustness.
high: selected from public-zero bisection evidence and existing public sources.
