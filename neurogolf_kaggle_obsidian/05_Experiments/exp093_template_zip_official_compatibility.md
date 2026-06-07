# exp093_template_zip_official_compatibility

## 目的

ユーザー提供 `C:\Users\doran\Downloads\neurogolf_templates.zip` のテンプレートが、現在の公式NeuroGolf utilsでそのまま `cost<=250〜600` の提出候補になり得るかを調査する。

## 結果

- tested templates: `7`
- official compatible: `0`

代表テンプレートはローカル `uint8 [H,W]` 実行では動くが、公式互換ではない。

## 理由

zip内テンプレートは `uint8 [H,W]` または `int32 [H,W]` の整数グリッドを想定している。入力名/出力名も `in` / `out`。一方、このworkspaceで使っている公式 `convert_to_numpy` / `verify_subset` は `FLOAT [1,10,30,30]` one-hot tensorを `input` / `output` として直接比較する。

そのため公式 `infer_static_ok` では `dynamic shape in` で落ち、公式runtimeでも `input` feedに対して `in` が無いというエラーになる。

## Decision

zipはdrop-in codeとしては不採用。設計思想、特に中間テンソルを減らす・1ノード化するという方向性は使う。ただし、整数grid wrapperを公式one-hot表現へ足すと、250〜600の利点は崩れやすい。

## Risk

- leakage risk: low。
- overfitting risk: low。
