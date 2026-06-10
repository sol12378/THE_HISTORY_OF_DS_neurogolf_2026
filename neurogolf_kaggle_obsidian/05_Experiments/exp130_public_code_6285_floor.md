# exp130_public_code_6285_floor

## Hypothesis

public CODEのbeicicc golf-domain artifactを提出候補へ昇格すれば、LB 6285を下限化できる可能性がある。

## Result

- source: `exp_b037_beicicc_golf_blend`
- source reference: `beicicc_lb_reference=6645`
- local static estimate: `7115.920621`
- source split: `beicicc_6645=379`, `exp_b025_best=21`
- zip sanity:
  - count: `400`
  - parse failure: `0`
  - too large: `0`
  - banned op: `0`
  - sha256: `ee5379dc10d30fcfc21ff99a921e802c38221aea1cb1c51145a003f726187be3`
- Kaggle submission ref: `53450559`
- current status: `COMPLETE`
- publicScore: `1400.37`

## Decision

6285 floor候補は棄却。public reference LB 6645とlocal static estimateは、このworkspaceで再構成したexact zipのKaggle LBへ転移しなかった。current submit-safe bestはexp_b025 LB `5930.40` のまま。

## Risk

public CODE由来なのでpublic LB overfit riskがある。private LB保証ではない。compiler/rule改善とは分け、提出下限確保用の候補として扱う。

## Interpretation

train/test validationとstatic costだけでgolf-domain public artifactを下限化するのは危険。今回の `1400.37` は、custom golf domainの互換性、hidden汎化、またはpublic CODE artifact再構成のどこかに重大な不一致があることを示す。以後、public CODEは直接提出ではなく、task別teacher・operator design・compiler priorとしてのみ使う。
