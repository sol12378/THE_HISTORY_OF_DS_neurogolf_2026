# exp073_task251_closed_component_shape_audit

## 目的

`task251` のclosed zero-component ruleを低cost化できるか、正例componentの形状を監査する。

## 結果

- positive components: `367`
- negative components: `641`
- positive rectangles: `241/367`
- positive boundary colors: `[2]` が `367/367`
- positive ray4_all: `367/367`
- positive adj_any_all: `367/367`
- negative boundary2 closed: `0`
- positive sizes: `8`, `6`, `4`, `10`, `12` のみ

## 解釈

正例はすべて「borderに接していない0成分」で、境界色は全て2。矩形は多いが全てではないため、単純なrectangle/bbox maskだけでは不十分。

本質は小ささや矩形性ではなく、border-connected 0成分を除外すること。

## 次アクション

既存artifactのreachability depthを削れるか試す。削れなければtask251は一旦保留し、task085 erase/mask cleanへ移る。
