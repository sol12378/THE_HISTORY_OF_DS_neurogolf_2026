# Leakage and Risks

- Avoid overfitting public examples.
- Separate diagnostic probes from trusted validation.
- Record notebook-only constraints and hidden-test assumptions.

## 2026-06-05 exp001

- データ取得は `data/external` に限定し、`data/raw` は変更しない。
- taskごとの公開 `train`, `test`, `arc-gen` は答えが見えているため、手書きONNXはpublic examplesへの完全一致を目的にしたものだと明記する。
- 公式評価にはprivate benchmark suiteも含まれるため、公開例全passは提出成功を保証しない。
- Public LBのみで複数taskを場当たり的に増やすと過適合的な運用になる。各taskで変換仮説、local pass/fail、cost、private不確実性を記録する。
