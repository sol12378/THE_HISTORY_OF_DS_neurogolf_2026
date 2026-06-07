# exp_b019_five_experiment_review_b014_b018

## 目的

user requirementに従い、exp_b014-b018の5実験が `cost<=250` / `7700` / LB最大化に本当に有意義だったかを問い直す。

## 判定

`meaningful_but_too_narrow_for_score`

## 有意義だった点

- task037 strict graph surgeryをfull-arc gate付きで棄却した。
- L1/L2上位に対して、単一object-role ruleと単一local predicateが弱いと確認した。
- L2 changed-cell profileからprecision 1.0のray feature islandを発見した。
- high-precision island ORだけではcoverage不足と確認した。

## 不十分だった点

- score deltaは `0.0` のまま。
- 提出候補なし。
- L2 sparse fillは診断が続き、低cost ONNX候補にまだ繋がっていない。

## Decision

次PDCAは一度、提出候補を作りやすいlaneへ寄せる。L2 findingsはpairwise-tree synthesisとして保存しつつ、次はL4 shape/cropや既存低cost artifact profileなど、より単純なloweringでlocal/LB較正候補を狙う。
