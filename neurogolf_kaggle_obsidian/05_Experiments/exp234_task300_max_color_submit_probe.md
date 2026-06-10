# exp234_task300_max_color_submit_probe

## 目的

exp178 current public bestに、exp233のtask300 rule candidateを1 taskだけ差し替えてKaggle提出する。

## 結果

- status: submitted complete
- ref: `53530035`
- public LB: `6006.32`
- task300 cost: `77546 -> 52653`
- expected local delta: `+0.387148`
- expected public LB if pass: `6006.32`
- zip sanity: 400 files, names ok

## 判断

期待値どおりpublic LBが改善したため、current public bestへ昇格する。

## リスク

- leakage risk: low for task300 delta
- private risk: medium because base exp178 includes high-risk source repairs
