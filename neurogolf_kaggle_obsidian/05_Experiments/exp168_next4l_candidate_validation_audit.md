# exp168_next4l_candidate_validation_audit

## 目的

exp167対象 `048/035/012/017` について、既存source候補のfull validation / costを事前監査する。

## 結果

- targets: `048/035/012/017`
- full_ok_total: `9`
- task048: full_ok `2`
- task035: full_ok `2`
- task012: full_ok `2`
- task017: full_ok `3`

## 判断

4 taskすべてにfull-local-valid既存候補はある。ただしexp167で4 taskすべてpublic aliveと判明したため、即repairではなく将来のcost/source分析用に保持する。

## 成果物

- `experiments/exp168_next4l_candidate_validation_audit/result.json`
- `experiments/exp168_next4l_candidate_validation_audit/notes.md`
- `experiments/exp168_next4l_candidate_validation_audit/next4l_candidate_audit.csv`

## リスク

- leakage risk: medium-to-high。既存public/teacher候補はlookup-likeの可能性がある。
- overfitting risk: medium-to-high。full local validationだけではpublic/private robustnessを保証しない。
