# exp262_current_rank221_320_fullarc_bypass_sweep

## 目的

exp261 current best bundleのcost rank221-320にも、1-pass full-arc bypass surgeryの上積みが残るかを確認する。

## 結果

- base: `experiments/exp261_exp260_bypass_bundle_submit_probe/submission.zip`
- checkpoint stage: `after_task273`
- generated_candidate_count: `1831`
- fullarc_selected_count: `17`
- selected_tasks: `30, 345, 124, 78, 153, 188, 329, 212, 50, 3, 254, 369, 180, 45, 248, 357, 273`
- local_delta: `+0.5498990065818337`

## 判断

task048付近で評価が不安定になり完走しなかったが、checkpointで保存された17件はfull-arc selectedであり、local estimateを更新した。rawを保存していなかったため、exp263で候補を再生成・full-arc replayしてbundle化する。

## Risk

- leakage risk: low-medium。current best artifactへのgraph surgeryであり、full-arc gateは通している。
- overfitting risk: medium-low。sample20だけでなくfull-arc selectedだが、source bundle自体のprivate robustness riskは残る。
