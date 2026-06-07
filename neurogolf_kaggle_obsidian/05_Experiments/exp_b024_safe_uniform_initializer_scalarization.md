# exp_b024_safe_uniform_initializer_scalarization

## 目的

`exp_b023` で抽出した Seddik-style `compress_uniform_initializers` を、strict seed に対して安全側に絞って適用する。狙いは大幅改善ではなく、public notebook由来のONNX surgery patternを full-arc gate と Kaggle micro-delta submission で較正すること。

## 結果

- base: `exp005_top_cost_rewrite_strict`
- screened tasks: `80`
- safe initializer rows: `16`
- generated candidates: `10`
- improved: `8`
- rejected: `2`
- improved tasks: `36, 51, 77, 110, 128, 284, 358, 383`
- local delta: `+0.02748525230556531`
- new local estimate: `6282.257713252306`
- submission ref: `53417752`
- public LB: `5929.92`

## 解釈

改善幅は小さいが、全採用候補は full-arc validation を通過している。2件の rejected candidate は `ScatterElements` の indices/updates shape mismatch で落ちたため、bundleには含めていない。

`exp_b024` のLB deltaはstrict seed `5929.89` から `+0.03` で、local delta `+0.027485` とほぼ一致した。`exp068` では同じSeddik系post-passをより広く `exp066` base に適用し、LB `5930.27` まで伸びた。したがって `exp_b024` は最大スコア更新というより、strict seed だけに狭く当てた場合の安定性確認として扱う。

## Risk

- Leakage risk: low-to-medium。公開notebook由来の手筋だが、採用はstrict seed上のfull-arc gateで決めている。
- Overfitting risk: medium。uniform scalarizationはbroadcast前提が壊れると危険なので、shape mismatch候補をrejectする必要がある。

## Decision

`submission.zip` をKaggle提出し、ref `53417752` がLB `5929.92` で完了した。今後のbundle生成ではSeddik-style post-passを標準後処理にするが、7700への主経路はこの小外科ではなく、明示rule/compiler replacementでcost 250〜600台を量産すること。
