# exp129_compiler_farm_core

## Hypothesis

NeuroGolf専用IRとcost extractorをONNX生成前に置けば、full-grid中間やScatterND/Where型の高cost候補を事前rejectでき、実験farmをscore-producing候補へ集中できる。

## Result

- status: `pipeline_core_ready`
- 6285 floor ready: `False`
- max observed Kaggle LB in registry: `1400.37`
- public CODEはregistryへ取り込んだが、6285 submit floorとしては未証明。full-arc validationとLB evidenceがあるものだけfloorに昇格する。
- smoke guardrailではcheap lane (`Gather`, `Slice+Pad`) とreject lane (`Where`, `ScatterND`) を分離できた。
- bundle ledger smoke accepted count: `1`

## Leakage / Overfitting Risk

公開CODEやblend artifactを直接best bundleへ加算するとpublic LB overfitとvalidation leakageのリスクがある。今回のpipelineではteacher/intelligenceとして登録し、公式互換validationとKaggle較正が揃うまで提出下限とはみなさない。

## Next

1. Kaggle public CODE notebook URL/sourceをregistryに追加し、claimed LB 6285候補を個別にfull-arc再検証する。
2. IR lowering pluginを `phase1_rewrite_utils.py` のcandidate evaluationへ接続する。
3. accepted candidate bundleにSeddik-style post-passとfocused surgeryを標準適用する。
