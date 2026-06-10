# exp266_dtype_memory_probe

## 目的

roadmap Phase Bの前提として、official costのmemory計上が中間tensor dtypeに比例するかを合成ONNXで確認する。

## 結果

- Cast/Identity/Add/NotNot 系の合成モデルを生成。
- input dtype/rankをfloat 4Dへ修正後、実行自体は通った。
- ただし全候補で `score_network` が `None` を返し、official memory/paramsを取得できなかった。

## 判断

このprobeだけではdtype比例計上のgo/no-goは判定できない。単純Cast/Identity/Addがscore対象として弱い可能性があるため、次は既存artifactの一部dtype変換、またはConv/Whereを含む実測済み型のprobeに切り替える。

## Risk

- leakage risk: none。合成score probeのみ。
- overfitting risk: none。
