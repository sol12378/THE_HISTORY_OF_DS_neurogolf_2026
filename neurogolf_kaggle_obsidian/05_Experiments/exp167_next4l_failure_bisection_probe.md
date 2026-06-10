# exp167_next4l_failure_bisection_probe

## 目的

exp166 current bestをベースに、残りsubset頻度上位の `048/035/012/017` をfail-stub化してpublic-zero有無を測る。

## 結果

- targets: `048/035/012/017`
- expected_drop_if_all_alive: `61.823685547451404`
- expected_lb_if_all_alive: `5906.356314452549`
- zip sha256: `ce1c4148006ffb73aeaa67e651f40c2df476ab53dd3c081c0c5f4269a3f50c85`
- Kaggle ref: `53523591`
- status: `COMPLETE`
- public LB: `5906.33`

## 判断

観測LB `5906.33` は all-alive期待 `5906.36` と一致。task048/035/012/017 はpublic-zeroではないため、短期repair suspectから除外する。

## 成果物

- `experiments/exp167_next4l_failure_bisection_probe/result.json`
- `experiments/exp167_next4l_failure_bisection_probe/notes.md`
- `experiments/exp167_next4l_failure_bisection_probe/submission.zip`

## リスク

- leakage risk: low。意図的fail-stub probe。
- overfitting risk: medium。提出枠を消費し、public scoring挙動のみを測る。
