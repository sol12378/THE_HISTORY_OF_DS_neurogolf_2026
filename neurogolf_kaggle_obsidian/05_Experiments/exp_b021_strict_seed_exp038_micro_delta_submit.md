# exp_b021_strict_seed_exp038_micro_delta_submit

## 目的

strict seedにexp038 full-arc-safe graph surgeryの4 taskだけを載せ、小さいlocal/LB較正提出候補を作る。

## 結果

- source strict seed: `exp005_top_cost_rewrite_strict`
- delta source: `exp038_fullarc_gated_noop_bypass`
- delta tasks: `62, 145, 255, 268`
- validation:
  - task062: `267_pass_0_fail`
  - task145: `267_pass_0_fail`
  - task255: `265_pass_0_fail`
  - task268: `266_pass_0_fail`
- local estimate delta: `+0.201867`
- new local estimate: `6282.432095`
- zip sanity: 400 files, names ok
- Kaggle submission ref: `53417203`
- Kaggle status: `PENDING` as of first check

## 解釈

exp041の大きなcollapseを避けるため、strict seedにfull-arc-safeな4 taskだけを載せたmicro deltaとして提出した。これは大幅score狙いではなく、local/LB対応を小さい差分で測るcalibration submissionである。

## Risk

- leakage risk: low-to-medium。strict seed baseで、deltaはfull-arc validated graph surgery。
- overfitting risk: medium。task-specific surgeryなので、LB deltaで確認する。

## Decision

LBが出たら `LB_Tracking.md` に反映し、micro deltaがLBで維持されるかを確認する。
