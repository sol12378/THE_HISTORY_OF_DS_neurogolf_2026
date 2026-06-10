# exp298_bypass_sweep_yield_analysis

## 目的

exp254以降の full-arc gated bypass sweep / bundle 提出系列を集計し、どこまで score-producing な graph surgery が残っているかを分析する。これは exp297 採点待ち中の独立作業であり、提出対象ではない。

## 集計

| source | selected | local delta | avg delta | 最大単発 |
|---|---:|---:|---:|---:|
| exp254 top30 | 13 | 0.090241 | 0.006942 | 0.022376 |
| exp256 rank31-80 | 16 | 0.331473 | 0.020717 | 0.295570 |
| exp258 rank81-140 | 23 | 0.219056 | 0.009524 | 0.050162 |
| exp260 rank141-220 | 27 | 0.840087 | 0.031114 | 0.198697 |
| exp262 partial rank221-320 | 17 | 0.549899 | 0.032347 | 0.188515 |
| exp264 rank321-400 | 8 | 0.597660 | 0.074707 | 0.220977 |
| exp297 fresh remainder | 3 | 0.055209 | 0.018403 | 0.036641 |

合計は 107 task / local delta `+2.683624221464793`。task 重複はなく、これまでの bypass bundle は小さいがほぼ較正通りに public LB へ転写されている。

## 解釈

- 今日の大きな前進は public-zero repair ではなく、既存 current best artifact の safe graph surgery を広い rank window に展開できたこと。
- exp254-265 は public LB `6005.93 -> 6008.90` を作った主因で、local/LB 転写が安定している。
- ただし exp297 の fresh は3件のみで、rank221-320 の残りを skip 完走しても追加deltaは `+0.0552`。同じ one-pass bypass supplier の辺際収益はかなり低下している。
- exp264 rank321-400 は低cost側なのに avg delta が最大だった。低cost artifact でも小さな node bypass が log score 上大きく効くため、低rank側を無視しない価値はある。

## 次アクション

1. exp297 の public LB を確認し、`6008.95` 付近なら current best として採用する。
2. bypass lane は完全停止ではなく、runtime-stopper task048/task336 の個別修復か、2-pass/greedy replay を狭く試す。ただし期待deltaは小さい。
3. 7000 へは bypass micro delta だけでは桁が足りない。次の score-producing 主戦場は roadmap Phase C の fused low-cost lowering / solved-rule 換金へ戻す。

## リスク

- leakage risk: low-medium。full-arc gated artifact surgery だが、既存 artifact の構造に依存する。
- overfitting risk: medium-low。これまで public LB 転写は良好だが、micro delta は表示丸めの影響を受けやすい。
