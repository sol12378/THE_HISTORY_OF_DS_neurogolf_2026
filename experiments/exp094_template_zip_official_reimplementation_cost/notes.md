# exp094_template_zip_official_reimplementation_cost

## 目的

zipテンプレートは直接公式互換ではないが、同じ幾何変換を公式one-hot形式で再実装した場合に250〜600級へ入るかを測る。

## 結果

`result.json` / `official_reimplementation_costs.csv` を参照。

## Decision

zipはdrop-in codeではなく、template taxonomyとして使う。純粋な `Slice` / `Transpose` / channel `Gather` 変換は公式one-hotでも安くなり得るが、uint8整数grid前提のcost値はそのまま移植できない。
