# exp320_task243_101_repair_source_audit

## 目的

exp319 で task243/task101 の fail-stub が Public LB を落とさなかったため、この2 task を public-zero repair target として扱い、既存 accepted candidate source を監査する。

## 仮説

既存 public-code/raw blend manifest には task243/task101 の full-local-valid 候補があり、full-arc gate を通せば repair zip として `+27.779018` 付近の public gain を狙える。

## 結果

- status: `completed`
- source manifest: `experiments/exp002_public_blend_6500_fast/candidate_manifest.csv`
- task101: accepted `24` rows。top は `franksunp_blended_best`、cost `65875`、local points `13.904485714156337`。
- task243: accepted `28` rows。top は `franksunp_blended_best`、cost `67878`、local points `13.874532744845478`。
- expected_public_gain_if_both_fixed: `27.779018459001815`
- expected_public_lb_if_both_fixed: `6036.739018459002`

## 判断

提出なし。manifest validation は `6/0` なので、次実験で実 candidate を抽出し、full-arc validation と zip sanity を通してから repair submit する。

## リスク

- leakage risk: medium-high。top 候補は public-code/raw blend source 由来で、private robustness は低めに見積もる。
- overfitting risk: medium。public diagnostic で target を特定しているため、採用判断は full local validation と expected LB 一致に限定する。
