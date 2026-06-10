# exp168_next4l_candidate_validation_audit

## 目的

exp167対象 `048/035/012/017` について、既存source候補のfull validation / costを事前監査する。

## 結果

- targets: `[48, 35, 12, 17]`
- full_ok_total: `9`

```json
[
  {
    "task_id": 48,
    "source_rows": 33,
    "unique_raw_candidates": 3,
    "full_ok_candidates": 2,
    "best_candidates": [
      {
        "task_id": "48",
        "filename": "task048.onnx",
        "source_label": "massimilianoghiotto_6254",
        "source_ref": "massimilianoghiotto/neurogolf2026-6254",
        "relative_path": "submission\\task048.onnx",
        "sha256": "9423c02033058a1188fae998353c04f6fbf4e48361dbee5461896ddd5d49ac28",
        "file_bytes": "4058",
        "normalized_bytes": "4683",
        "params": "66",
        "memory_bytes": "5543",
        "cost": "5609",
        "simple_cost": "4749",
        "local_points": "16.367872270491663",
        "validation_status": "270_pass_0_fail",
        "validation_pass": "10",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "5609",
        "raw_sha256": "010a8d8bad3f868c1dc1b32f62929c87cb98b880cf291e9ad90cbc8ab094099a",
        "raw_path": "experiments\\exp168_next4l_candidate_validation_audit\\task048_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 5609,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "48",
        "cost": "5609",
        "points": "16.367872",
        "exp": "exp_b035_new_source_full_arc_blend",
        "manifest": "experiments\\exp_b035_new_source_full_arc_blend\\selected_manifest.csv",
        "manifest_cost": "5609",
        "raw_sha256": "9423c02033058a1188fae998353c04f6fbf4e48361dbee5461896ddd5d49ac28",
        "raw_path": "experiments\\exp168_next4l_candidate_validation_audit\\task048_03_exp_b035_new_source_full_arc_blend.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "validation_status": "270_pass_0_fail",
        "official_cost": 5609,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  },
  {
    "task_id": 35,
    "source_rows": 33,
    "unique_raw_candidates": 5,
    "full_ok_candidates": 2,
    "best_candidates": [
      {
        "task_id": "35",
        "filename": "task035.onnx",
        "source_label": "massimilianoghiotto_6254",
        "source_ref": "massimilianoghiotto/neurogolf2026-6254",
        "relative_path": "submission\\task035.onnx",
        "sha256": "15a9c474e70c83253daaa3ff80387d3b29feadebd4ec8018b87fc277cf69f62a",
        "file_bytes": "3707",
        "normalized_bytes": "5343",
        "params": "51",
        "memory_bytes": "15312",
        "cost": "15363",
        "simple_cost": "5394",
        "local_points": "15.36028269986934",
        "validation_status": "266_pass_0_fail",
        "validation_pass": "6",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "15363",
        "raw_sha256": "415e6719cce74fd480f7999d3bd42e52a3974539fbad353c91d13ee86dcfd930",
        "raw_path": "experiments\\exp168_next4l_candidate_validation_audit\\task035_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 15363,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "35",
        "cost": "15363",
        "points": "15.360283",
        "exp": "exp_b035_new_source_full_arc_blend",
        "manifest": "experiments\\exp_b035_new_source_full_arc_blend\\selected_manifest.csv",
        "manifest_cost": "15363",
        "raw_sha256": "15a9c474e70c83253daaa3ff80387d3b29feadebd4ec8018b87fc277cf69f62a",
        "raw_path": "experiments\\exp168_next4l_candidate_validation_audit\\task035_05_exp_b035_new_source_full_arc_blend.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "validation_status": "266_pass_0_fail",
        "official_cost": 15363,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  },
  {
    "task_id": 12,
    "source_rows": 33,
    "unique_raw_candidates": 5,
    "full_ok_candidates": 2,
    "best_candidates": [
      {
        "task_id": "12",
        "filename": "task012.onnx",
        "source_label": "massimilianoghiotto_6254",
        "source_ref": "massimilianoghiotto/neurogolf2026-6254",
        "relative_path": "submission\\task012.onnx",
        "sha256": "6565dbdc01b75cf0fd11dff727c2312655bcb5fb9e59a8020020ef04a7b5a80c",
        "file_bytes": "1951",
        "normalized_bytes": "2599",
        "params": "103",
        "memory_bytes": "12536",
        "cost": "12639",
        "simple_cost": "2702",
        "local_points": "15.555457449352874",
        "validation_status": "265_pass_0_fail",
        "validation_pass": "5",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "12639",
        "raw_sha256": "c169f12ce3baf4c253e1d7773020c00f99fdaf1de0576dcfb8378541558f1909",
        "raw_path": "experiments\\exp168_next4l_candidate_validation_audit\\task012_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 12639,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "12",
        "cost": "12639",
        "points": "15.555457",
        "exp": "exp_b035_new_source_full_arc_blend",
        "manifest": "experiments\\exp_b035_new_source_full_arc_blend\\selected_manifest.csv",
        "manifest_cost": "12639",
        "raw_sha256": "6565dbdc01b75cf0fd11dff727c2312655bcb5fb9e59a8020020ef04a7b5a80c",
        "raw_path": "experiments\\exp168_next4l_candidate_validation_audit\\task012_05_exp_b035_new_source_full_arc_blend.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "validation_status": "265_pass_0_fail",
        "official_cost": 12639,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  },
  {
    "task_id": 17,
    "source_rows": 33,
    "unique_raw_candidates": 6,
    "full_ok_candidates": 3,
    "best_candidates": [
      {
        "task_id": "17",
        "source": "exp068_seddik_style_strict_scalarization_seddik_style_scalarize_dedup_prune",
        "template_name": "seddik_style_scalarize_dedup_prune",
        "route": "sparse_edit_or_object_completion",
        "cost": "34889",
        "local_points": "14.540073127737525",
        "file_bytes": "10120",
        "status": "improved",
        "reason": "ok",
        "sha256": "c980016edb94eb43babb9e711f24aa398b008229d464194bee443dd089850945",
        "exp": "exp068_seddik_style_strict_scalarization",
        "manifest": "experiments\\exp068_seddik_style_strict_scalarization\\selected_manifest.csv",
        "manifest_cost": "34889",
        "raw_sha256": "c980016edb94eb43babb9e711f24aa398b008229d464194bee443dd089850945",
        "raw_path": "experiments\\exp168_next4l_candidate_validation_audit\\task017_04_exp068_seddik_style_strict_scalarization.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "validation_status": "266_pass_0_fail",
        "official_cost": 34889,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "17",
        "cost": "34889",
        "points": "14.540073",
        "exp": "exp_b035_new_source_full_arc_blend",
        "manifest": "experiments\\exp_b035_new_source_full_arc_blend\\selected_manifest.csv",
        "manifest_cost": "34889",
        "raw_sha256": "5d93fd369fc7f1bab688d7b3d2b5dd12613c057f2b19afffbf609c283d3bfc2f",
        "raw_path": "experiments\\exp168_next4l_candidate_validation_audit\\task017_06_exp_b035_new_source_full_arc_blend.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "validation_status": "266_pass_0_fail",
        "official_cost": 34889,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "17",
        "filename": "task017.onnx",
        "source_label": "massimilianoghiotto_6254",
        "source_ref": "massimilianoghiotto/neurogolf2026-6254",
        "relative_path": "submission\\task017.onnx",
        "sha256": "fb2137630963bd757e3fa76d251d504d01fbf7b1b6106c0e123e2b4b6247e574",
        "file_bytes": "8855",
        "normalized_bytes": "10570",
        "params": "644",
        "memory_bytes": "34685",
        "cost": "35329",
        "simple_cost": "11214",
        "local_points": "14.527540564604424",
        "validation_status": "266_pass_0_fail",
        "validation_pass": "6",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "35329",
        "raw_sha256": "d1f54d1f58e001a1eede31fda2a8e667d2f2e0fdf4323f0a00a1dcb18187c52d",
        "raw_path": "experiments\\exp168_next4l_candidate_validation_audit\\task017_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 35329,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  }
]
```

## 判断

If exp167 identifies a public-zero target, use this audit to pick any distinct full_ok repair candidate before rule mining.

## リスク

medium-to-high: existing public/teacher candidates may be lookup-like.
medium-to-high: full local validation is not sufficient for public/private robustness.
