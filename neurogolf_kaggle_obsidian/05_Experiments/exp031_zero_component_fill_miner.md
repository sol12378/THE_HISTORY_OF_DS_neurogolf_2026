# exp031_zero_component_fill_miner

## Hypothesis

入力の非ゼロセルを壁とみなし、0 connected componentのうち外周に接続する成分を外側色、閉じた成分を内側色で塗れば、region/line fill系の上位taskを説明できる。

## Result

- target tasks: `198, 187, 203, 313`
- arc-gen sample: `20`
- pass tasks: `187`
- task187 rule: border-connected zero component -> `3`, enclosed zero component -> `2`, nonzero wall cells preserve input color
- task187 validation: train `3/3`, test `1/1`, arc-gen sample20 `20/20`
- local estimate delta: `0.0`
- submission: no submit, ONNX未生成

## Interpretation

task187の規則は説明可能で、train/test/sample20に通った。
ただしexp019でnaive unrolled flood-fill ONNXはcost `237631+` となりbaseline `105313` に負けている。
次は同じ規則を、閉形式のrow/column prefix mask、既存artifact surgery、または別のcost-aware loweringで圧縮できるかを検証する。

## Risks

- leakage risk: low-medium。色はtrainのみで推定し、test/arc-gen sample20は検証に使った。
- overfitting risk: medium。sample20 passはprivate-like holdoutではない。

## Decision

task187は規則探索を完了扱いにし、次PDCAではONNX loweringだけを問題として扱う。naive flood-fill unrollは再採用しない。
