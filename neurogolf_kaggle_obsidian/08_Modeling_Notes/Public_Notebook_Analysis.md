# Public Notebook Analysis

## 2026-06-05

対象:

- `needless090/neurogolf-4250`
- `magmacot/neurogolf-new-blending`
- `karnakbaevarthur/all-task-description-analysis`
- `vyankteshdwivedi/neurogolf-multi-source-onnx-solver`

## 要約

- `vyankteshdwivedi/neurogolf-multi-source-onnx-solver` が主力候補。多数のpublic artifact/sourceからtask別ONNX候補を集め、static shape filter、source priority、自作solver、artifact lock、graph rewriteを組み合わせる。
- `magmacot/neurogolf-new-blending` は薄いblend wrapper。attached notebookの `submission.zip` を集め、input/output rename後に `params + serialized bytes` の簡易costでtask別に最小候補を選ぶ。公式memory costではないため採用時は再検証が必要。
- `needless090/neurogolf-4250` はsolverではなく、外部ONNX bundleを deterministic に再zipするだけ。scoreの本体はattached artifact側。
- `karnakbaevarthur/all-task-description-analysis` はDeepSeekによるtask primitive分類と、複数zipのvalidate/merge例。solver実装というよりtask triageの地図として有用。

## 6500へ向けた示唆

- 6254級artifact lockを土台に、task別に公式validationと公式costで候補を選び直す。
- public notebookの簡易costやfile sizeだけではなく、`neurogolf_utils` 相当でtrain/test/arc-gen passとmemory+paramsを確認する。
- 追加sourceを増やす場合は、task別採用表を作り、各taskのsource、cost、local pass/fail、LB差分を記録する。
- 低complexity task分類を使って、単純幾何変換、color mapping、crop/tile、symmetry系から自前改善を進める。

## リスク

- public artifact blendはpublic LB過適合・再現性・由来不明ONNXのリスクがある。
- `vyanktesh` notebookには通常実行で公式再検証がoffになっている箇所があり、そのまま信用しすぎない。
- `magmacot` のcostは公式cost proxyではない。
- `needless090` はsource選択が `/kaggle/input` の探索順に依存する。
