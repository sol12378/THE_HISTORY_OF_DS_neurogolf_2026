# exp093_template_zip_official_compatibility

## 目的

`C:\Users\doran\Downloads\neurogolf_templates.zip` のテンプレートが、現在の公式NeuroGolf utilsでそのまま `cost<=250〜600` の提出候補になり得るかを調査する。

## 結果

- tested templates: `7`
- official compatible: `0`

## 解釈

zip内テンプレートは `uint8 [H,W]` または `int32 [H,W]` の整数グリッドを想定している。一方、現在の公式 `convert_to_numpy` / `verify_subset` は `FLOAT [1,10,30,30]` one-hot tensorを直接比較する。入出力名もzip側は `in`/`out`、公式側は `input`/`output` を前提にしている。

したがって、zipテンプレートの低cost値はこのworkspaceの公式評価契約とは直接一致しない。使うなら、個別taskでone-hot公式表現へ移植し、公式 `score_network` で再計測する必要がある。

## Decision

直接提出候補としては不採用。設計思想、特に「中間テンソル数を減らす」「uint8/整数グリッドなら安い」という方向性は参考にする。ただしwrapper変換を足すと250〜600の利点は崩れる可能性が高い。

## Risk

- leakage risk: low。
- overfitting risk: low。
