# exp073_task251_closed_component_shape_audit

## 目的

`task251` のclosed zero-component ruleを低cost化できるか、正例componentの形状を監査する。

## 結果

- positive components: 367
- negative components: 641
- positive rectangles: 241/367
- positive bbox shapes: [('2x2', 56), ('3x3', 52), ('2x3', 52), ('3x2', 49), ('2x4', 45), ('4x2', 39), ('4x4', 25), ('3x4', 25), ('4x3', 24)]
- positive boundary colors: [('[2]', 367)]

## 判断

矩形性やbbox shapeの偏りが強ければ、flood-fill unrollではなくrectangle/bbox mask loweringへ進む。弱ければcomponent connectivityを別の安い方法で表現する必要がある。
