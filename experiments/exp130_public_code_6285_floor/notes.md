# exp130_public_code_6285_floor

## Hypothesis

public CODEのbeicicc golf-domain artifactを提出候補へ昇格すれば、LB 6285を下限化できる可能性がある。

## Result

- source: `exp_b037_beicicc_golf_blend`
- status: `submission_floor_candidate_ready`
- zip: `E:\kaggle\neurogolf-2026\experiments\exp130_public_code_6285_floor\submission.zip`
- sha256: `ee5379dc10d30fcfc21ff99a921e802c38221aea1cb1c51145a003f726187be3`
- names_ok: `True` / count `400`
- parse_failure_count: `0`
- too_large_count: `0`
- banned_count: `0`
- manifest source_counts: `{'beicicc_6645': 379, 'exp_b025_best': 21}`
- Kaggle submission ref: `53450559`
- Kaggle status: `COMPLETE`
- Kaggle publicScore: `1400.37`

## Interpretation

6285 floor仮説は棄却。public reference LB 6645を根拠に提出したが、このworkspaceで作成したexact zipはKaggle publicScore `1400.37` に崩壊した。golf domain artifactはtrain/test validationと静的costでは良く見えるが、こちらの提出環境・hidden評価・またはpublic CODE再構成との互換性が成立していない。

## Leakage / Overfitting Risk

public CODE由来であり、public LB overfit riskがある。private performanceを保証しない。採用はLB下限確保用の候補として扱い、rule/compiler本体の改善とは分ける。

## Decision

このbundleはbest/floorに採用しない。current submit-safe bestはexp_b025のLB `5930.40` に戻す。public CODEは引き続きteacher/intelligence扱いに限定し、直接提出下限としては使わない。
