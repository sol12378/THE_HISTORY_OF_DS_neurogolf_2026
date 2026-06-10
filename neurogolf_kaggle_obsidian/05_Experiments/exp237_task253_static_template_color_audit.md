# exp237_task253_static_template_color_audit

## 目的

task253は4x4固定binary templateに見えるため、出力色が入力の簡単な特徴で決まるか監査する。

## 結果

- baseline cost: `48269`
- binary_signature_count: `1`
- best color selector:
  - `least_nz`: `76/265`
  - `mode_nz`: `76/265`

## 判断

static template自体は有望だが、色selectorが未解決。単純な集約特徴では提出候補にならない。

## リスク

- leakage risk: low
- overfitting risk: medium-low
