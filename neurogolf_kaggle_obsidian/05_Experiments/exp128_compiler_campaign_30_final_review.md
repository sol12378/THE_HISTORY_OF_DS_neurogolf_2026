# exp128_compiler_campaign_30_final_review

## Purpose

30実験compiler campaignの最終レビュー。30実験に達したため停止する。

## Result

- initial local: `6282.812218`
- final local: `6282.965236`
- total delta: `+0.153018`
- LB7500: not reached
- best final bundle candidate: `experiments/exp127_focused_surgery_seventh_pass_limited/submission.zip`

## Interpretation

score-producingだったのはfull-validation-gated focused graph surgeryのみ。これはLB calibrationとして有効だが、全400 taskをcost `250〜600`級へ落とすcompilerには届かなかった。

fresh loweringでは、selector、writeback、full-grid intermediateのcostが支配的だった。既存artifact harvestingも、final-shapeに見える中間tensorが最終outputの意味を保持していなかった。

## Decision

このcampaignは停止。再開するなら、ONNXを直接作る前にNeuroGolf専用IRとcost extractorを作る。
