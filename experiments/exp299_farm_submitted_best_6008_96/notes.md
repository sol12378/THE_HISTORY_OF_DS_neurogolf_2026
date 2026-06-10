# exp129_compiler_farm_core

## Hypothesis

NeuroGolf専用IRとcost extractorをONNX生成前に置けば、`params + output tensor bytes` のlocal estimateで候補を事前評価でき、実験farmをscore-producing候補へ集中できる。

## Result

- status: `pipeline_core_ready`
- 6285 floor ready: `False`
- max observed Kaggle LB in registry: `6008.96`
- public CODEはregistryへ取り込んだが、6285 submit floorとしては未証明。full-arc validationとLB evidenceがあるものだけfloorに昇格する。
- smoke guardrailではcandidate costをoutput tensor bytes中心に表示し、`Where` はMAC=0として扱う。`FULL_GRID_COMPOSITION` / `ScatterND` などの構造リスクは別guardrailで維持する。
- bundle ledger smoke accepted count: `1`
- bundle ledger smoke best-by-task count: `1`
- bundle ledger smoke duplicate task count: `0`
- submit-gate local delta uses best-total value: `0.6931471805599453`
- submitted best estimate: `6008.96`
- candidate estimate: `6009.65314718056`
- should submit: `True`
- submission decision scope: `smoke_dummy_not_kaggle_candidate`
- submission decision reason: `run_smoke uses synthetic ledger rows only; do not submit from this decision`

## Leakage / Overfitting Risk

公開CODEやblend artifactを直接best bundleへ加算するとpublic LB overfitとvalidation leakageのリスクがある。今回のpipelineではteacher/intelligenceとして登録し、公式互換validationとKaggle較正が揃うまで提出下限とはみなさない。

## Next

1. Kaggle public CODE notebook URL/sourceをregistryに追加し、claimed LB 6285候補を個別にfull-arc再検証する。
2. IR lowering pluginを `phase1_rewrite_utils.py` のcandidate evaluationへ接続する。
3. accepted candidate bundleにSeddik-style post-passとfocused surgeryを標準適用する。
