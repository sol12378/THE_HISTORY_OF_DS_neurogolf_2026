# exp294_bundle_full_arc_pass_gate

## Plan

real candidate manifest 生成へ戻る前に、accuracy gate の表記揺れを潰す。

既存 manifest は `validation_status=full_arc_pass` を使うことがあり、`from_selected_manifest()` 経由では `266_pass_0_fail` に正規化される。一方、`BundleCandidate` を直接作る real supplier では `full_arc_pass` のままだと accepted されないリスクがある。

見込み:

- `full_arc_pass` と `*_pass_0_fail` を accepted として扱う。
- `*_pass_1_fail` は rejected のまま。

## Do

`experiments/neurogolf_farm/bundle_manager.py` のみ変更。

- `BundleCandidate.accepted` に `accuracy_ok` を追加。
- `validation_status == "full_arc_pass"` または `validation_status.endswith("_pass_0_fail")` を accuracy gate pass とする。

## Check

synthetic probe:

```json
{
  "full_arc_pass_accepted": true,
  "pass0_accepted": true,
  "fail_accepted": false,
  "best_total_local_delta": 0.3285040669720361
}
```

## Act

No submit。synthetic accuracy-gate probe のみで、実候補bundleの local estimate 更新ではない。

## Risk

- Leakage risk: なし
- Overfitting risk: なし
- Operational risk: `full_arc_pass` の意味は既存 sweep の full-arc validation pass に限定して使う。
