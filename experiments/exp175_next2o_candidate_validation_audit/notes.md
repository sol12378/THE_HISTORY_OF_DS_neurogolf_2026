# exp175_next2o_candidate_validation_audit

## 目的

exp174対象 `285/286` について、既存source候補のfull validation / costを事前監査する。

## 結果

- targets: `[285, 286]`
- full_ok_total: `4`

```json
[
  {
    "task_id": 285,
    "source_rows": 34,
    "unique_raw_candidates": 6,
    "full_ok_candidates": 2,
    "best_candidates": [
      {
        "task_id": "285",
        "filename": "task285.onnx",
        "source_label": "kojimar_5800_minimal_blend",
        "source_ref": "kojimar/neurogolf-5800-55-minimal-onnx-blend-assets",
        "relative_path": "base_submission\\task285.onnx",
        "sha256": "d70e0ed55a67f9f23a5c100a505d7587b9739ff0ec31e4df49d5db7f657ff42b",
        "file_bytes": "98661",
        "normalized_bytes": "99305",
        "params": "24355",
        "memory_bytes": "127641",
        "cost": "151996",
        "simple_cost": "123660",
        "local_points": "13.068390516307327",
        "validation_status": "265_pass_0_fail",
        "validation_pass": "6",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "151996",
        "raw_sha256": "39f736e972e2cead06c1af94abca655ea9fb60e6fb10d8296b928174042b1d03",
        "raw_path": "experiments\\exp175_next2o_candidate_validation_audit\\task285_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 151996,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "285",
        "cost": "395468",
        "points": "12.112175",
        "exp": "exp_b035_new_source_full_arc_blend",
        "manifest": "experiments\\exp_b035_new_source_full_arc_blend\\selected_manifest.csv",
        "manifest_cost": "395468",
        "raw_sha256": "a7b6460a76896e8eabafd5d99bbac88a6a49d21bcf62107026bdf33a4e0f0c22",
        "raw_path": "experiments\\exp175_next2o_candidate_validation_audit\\task285_06_exp_b035_new_source_full_arc_blend.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "validation_status": "265_pass_0_fail",
        "official_cost": 395468,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  },
  {
    "task_id": 286,
    "source_rows": 33,
    "unique_raw_candidates": 4,
    "full_ok_candidates": 2,
    "best_candidates": [
      {
        "task_id": "286",
        "source": "massimilianoghiotto_6254",
        "baseline_cost": "162200",
        "new_cost": "162199",
        "baseline_points": "13.00341457933655",
        "new_points": "13.00342074458367",
        "removed_initializers": "1",
        "baseline_file_bytes": "24340",
        "new_file_bytes": "24315",
        "status": "improved",
        "reason": "ok",
        "exp": "exp005_top_cost_rewrite_strict",
        "manifest": "experiments\\exp005_top_cost_rewrite_strict\\rewrite_manifest.csv",
        "manifest_cost": "162199",
        "raw_sha256": "af22a897ae56ce01c2a97f72718e95b8faa9d7a2bad2bb839b6b63c4960c6b1d",
        "raw_path": "experiments\\exp175_next2o_candidate_validation_audit\\task286_02_exp005_top_cost_rewrite_strict.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "validation_status": "265_pass_0_fail",
        "official_cost": 162199,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "286",
        "filename": "task286.onnx",
        "source_label": "massimilianoghiotto_6254",
        "source_ref": "massimilianoghiotto/neurogolf2026-6254",
        "relative_path": "submission\\task286.onnx",
        "sha256": "4f012dc2e0a2b7b30e956f16edf2b883865d3a35d150e621bde7331b202015d5",
        "file_bytes": "19743",
        "normalized_bytes": "24340",
        "params": "1305",
        "memory_bytes": "160895",
        "cost": "162200",
        "simple_cost": "25645",
        "local_points": "13.00341457933655",
        "validation_status": "265_pass_0_fail",
        "validation_pass": "5",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "162200",
        "raw_sha256": "29abf4a27de955bab9fe9c99111e525a5123191264f3611a0c9abe10ec9f5694",
        "raw_path": "experiments\\exp175_next2o_candidate_validation_audit\\task286_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 162200,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  }
]
```

## 判断

If exp174 identifies a public-zero target, use this audit to pick a distinct full_ok repair candidate before rule mining.

## リスク

medium-to-high: existing public/teacher candidates may be lookup-like.
medium-to-high: full local validation is not sufficient for public/private robustness.
