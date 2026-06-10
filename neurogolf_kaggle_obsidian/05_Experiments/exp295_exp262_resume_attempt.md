# exp295_exp262_resume_attempt

## Plan

exp262 rank221-320 sweep の partial checkpoint (`after_task273`) 以降に、未提出の追加selectedが残っているか確認する。

## Do

既存 `run_exp262.py` を再実行。

## Check

- exit code: `1`
- checkpoint: `after_task273`
- selected: `17`
- local_delta: `0.5498990065818337`
- fresh delta vs current best: `0.0`
- failure: `task048` 付近の ONNXRuntime reshape/conv runtime error

## Act

提出なし。追加selectedなし。

## Next

task048 bad candidatesをskip/quarantineして、rank221-320の残りを完走する。
