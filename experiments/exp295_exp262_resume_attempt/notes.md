# exp295_exp262_resume_attempt

## Plan

Phase C の score-producing候補生成へ戻るため、partialで止まっていた `exp262_current_rank221_320_fullarc_bypass_sweep` を既存scriptで再実行し、`after_task273` 以降に未提出の追加selectedが出るか確認する。

見込み:

- exp262 partial は `+0.549899` / 17 selected まで exp263 で提出済み。
- `task273` 以降に full-arc pass の追加候補があれば、現行best `6008.90` への fresh local estimate 更新になり得る。

## Do

既存scriptを実行:

```powershell
.\.venv\Scripts\python.exe experiments/exp262_current_rank221_320_fullarc_bypass_sweep/run_exp262.py
```

## Check

結果:

- exit code: `1`
- checkpoint: `after_task273`
- generated_candidate_count: `1831`
- fullarc_selected_count: `17`
- selected_tasks: `30/345/124/78/153/188/329/212/50/3/254/369/180/45/248/357/273`
- local_delta: `0.5498990065818337`

これは既に exp263 で提出済みの partial checkpoint と同じ内容で、fresh候補は増えていない。

失敗:

- `task048` 付近で ONNXRuntime の reshape/conv runtime error が出て script が非ゼロ終了。
- rank221-320 の残りを完走できなかった。

## Act

No submit。fresh local estimate 改善は `0.0`。

## Next

`task048` の bad candidate を skip/quarantine して、rank221-320 の残りを完走できるようにする。候補生成script本体を触る場合は、次ループの制約と相談する必要がある。

## Risk

- Leakage risk: なし
- Overfitting risk: なし
- Operational risk: exp262 partial は現行best lineageに含まれるため、同じ17件を再提出しない。
