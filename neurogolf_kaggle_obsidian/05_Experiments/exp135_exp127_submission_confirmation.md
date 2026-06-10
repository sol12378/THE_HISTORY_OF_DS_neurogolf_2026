# exp135_exp127_submission_confirmation

## 目的

`docs/experiment_plan_2026-06-10.md` の Phase 0 を実行し、exp127 working bundle の提出結果を確認する。

## 結果

- exp127 はすでに `2026-06-08` に Kaggle 提出済み。
- ref: `53480507`
- public LB: `5930.55`
- source local: `6282.965236`
- exp_b025 から LB `+0.15`、local `+0.153`。
- gap は約 `-352.42` で維持。

## 解釈

micro delta はまた LB へほぼ転写した。したがって、cost 計測や正解 task の点数転写は安定しており、残る大きな gap は private functional failure task 群で説明するのが最も自然。

## 判断

重複提出はしない。Phase 0 完了。次は Phase A の failure task 特定を優先する。

## リスク

- leakage risk: low。
- overfitting risk: low-to-medium。
