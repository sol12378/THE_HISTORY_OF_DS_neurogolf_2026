# exp002_public_blend_6500_fast notes

## 仮説

公開artifact/sourceをtask別に収集し、`massimilianoghiotto/neurogolf2026-6254` をbaseに、static validationとofficial-like costで安い候補へ置換すれば、まずlocal推定6500台の `submission.zip` を作れる。

## 実行条件

- 実行日: 2026-06-05
- 評価: `train + test + arc-gen[:2]`
- cost: official utility互換の `memory bytes + params`
- local points: `max(1, 25 - ln(cost))`
- 必須filter:
  - ONNX parse/check/shape inference pass
  - file size <= 1.44MB
  - input/output正規化
  - banned ops: `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`, `Compress`
  - dynamic shapeなし
  - sample validation failはreject

## 結果

- status: `local_estimate_below_target`
- local estimate: `6282.075718068686`
- selected tasks: `400 / 400`
- candidate count: `16346`
- accepted candidates: `13609`
- validation: selected `400 pass`
- zip sanity: `submission.zip` は `task001.onnx` ... `task400.onnx` の400ファイルのみ。

## 採用source

- `massimilianoghiotto_6254`: 329 tasks
- `afr1ste_5689_artifact`: 44 tasks
- `kojimar_5800_minimal_blend`: 14 tasks
- `franksunp_blended_best`: 9 tasks
- `vyanktesh_multi_source_output`: 2 tasks
- `needless090_4250_output`: 2 tasks

## 主なreject理由

- `banned op Compress`: 1702
- `bad dim input`: 465
- `functions are not allowed`: 318
- `dynamic shape cr_x`: 36
- `sample validation failed`: 21

高スコア公開artifactとして `beicicc/neurogolf-6645-39-open-submission-artifact` や6335/6323/6285系も取得したが、今回のstatic/banned-op基準では多くがrejectされ、採用改善はなかった。追加kernel outputも同様で、local estimateは `6282.08` で頭打ちになった。

## Leakage / Overfitting Risk

- leakage risk: 高。public artifact/sourceを直接利用した最速LB寄せであり、private benchmarkやrule updateに弱い。
- overfitting risk: 高。公開LB実績のあるONNX bundleをtask別にblendする方針で、汎化性能の検証ではない。

## 次PDCA

1. official rulesを再確認し、`Compress` / model functions / dynamic shapeの扱いが本当に禁止かを再監査する。
2. `task366`, `task133`, `task158`, `task173`, `task286` などtop expensive tasksから、自前templateまたは安全なrewriteでcostを下げる。
3. 6500台を狙うには、現行strict filterのままでは公開artifact追加だけでは不足。cost rewriteまたはルール許容範囲の再解釈が必要。
4. Kaggle submit前には、採用済み400 taskでfull `arc-gen` validationへ拡張する。
