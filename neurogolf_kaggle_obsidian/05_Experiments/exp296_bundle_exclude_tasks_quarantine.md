# exp296_bundle_exclude_tasks_quarantine

## Plan

task048 のような既知runtime-stopper taskを fresh submit review から除外できるようにする。

## Do

`bundle_manager.py` に `exclude_tasks` を追加。

## Check

- no quarantine: 2 accepted, `best_total_local_delta=0.9162907318741551`
- task048 quarantine: 1 accepted, `best_total_local_delta=0.22314355131420976`
- task048+task049 quarantine: 0 accepted, `should_submit=false`

## Act

提出なし。synthetic probe のみ。
