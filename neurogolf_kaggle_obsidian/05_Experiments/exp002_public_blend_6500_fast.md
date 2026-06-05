# exp002_public_blend_6500_fast

## 目的

公開artifact/sourceを集め、`massimilianoghiotto/neurogolf2026-6254` をbaseにtask別最小cost候補へ置換し、local推定6500台の `submission.zip` を作る。

## 仮説

公開artifactの中には6254 bundleより低costで同一taskを解けるONNXが十分含まれており、strict static validationを通してblendすれば6500台へ到達できる。

## 結果

- local estimate: `6282.075718068686`
- status: `local_estimate_below_target`
- selected: `400 / 400`
- candidate count: `16346`
- accepted candidates: `13609`
- selected validation: `400 pass`
- Kaggle submit: なし

## 採用source

- `massimilianoghiotto_6254`: 329
- `afr1ste_5689_artifact`: 44
- `kojimar_5800_minimal_blend`: 14
- `franksunp_blended_best`: 9
- `vyanktesh_multi_source_output`: 2
- `needless090_4250_output`: 2

## 解釈

6500台には未達。高スコア公開artifactを追加しても、今回の基準では `Compress`, model functions, dynamic shape, invalid dim で大量にrejectされ、採用可能な改善は限定的だった。

現行strict filterのまま公開artifactを足すだけでは伸びが鈍い。次はofficial rules上の禁止/許容範囲を再監査し、そのうえでtop expensive tasksの自前rewriteを行う必要がある。

## Risk

- leakage risk: 高。public artifact/sourceを直接利用している。
- overfitting risk: 高。public LB寄せであり、private汎化を保証しない。

## 次PDCA

- `Compress` / functions / dynamic shapeの扱いをofficial specで再確認。
- `task366`, `task133`, `task158`, `task173`, `task286` からcost rewrite。
- Kaggle submit前にfull `arc-gen` validationへ拡張。
