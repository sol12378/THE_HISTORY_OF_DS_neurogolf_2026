# exp136_private_failure_subset_inventory

## Hypothesis

local/LB gap `-352.4` は、exp_b025/exp127 の cost 計測差ではなく、約24 taskのprivate functional failureで説明できる。

## Result

- target_gap: `352.412218`
- task inventory rows: `400`
- subset candidates within tolerance: `50`
- best subset k: `24`
- best subset sum_points: `352.41`
- best subset gap_error: `0.0`
- best subset tasks: `002 004 008 009 013 014 016 018 019 021 023 024 025 029 031 032 034 036 046 047 049 050 051 366`

## Interpretation

固定配布の `arc-gen` だけではprivate failureを直接再現できないが、per-task scoreのsubset-sumから、24 task前後の失敗集合でgapを説明できることを確認した。これは計画書のA仮説と整合する。

次は上位subset候補をsource/route別に分割し、bisection probeを使う前に、各taskのモデルprovenanceとshape外挿リスクを個別監査する。

## Leakage / Overfitting Risk

診断のみで提出物は作らない。subset-sumは状況証拠であり、private failure taskを証明するものではない。
