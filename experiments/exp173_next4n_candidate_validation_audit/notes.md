# exp173_next4n_candidate_validation_audit

## 目的

exp171対象 `003/038/001/086` について、既存source候補のfull validation / costを事前監査する。

## 結果

- targets: `[3, 38, 1, 86]`
- full_ok_total: `8`

```json
[
  {
    "task_id": 3,
    "source_rows": 33,
    "unique_raw_candidates": 2,
    "full_ok_candidates": 1,
    "best_candidates": [
      {
        "task_id": "3",
        "filename": "task003.onnx",
        "source_label": "afr1ste_5689_artifact",
        "source_ref": "afr1ste/neurogolf-5689-51-current-rules-open-artifact",
        "relative_path": "submission\\task003.onnx",
        "sha256": "af63acf28b90fea42960fbbf559775a83a53e0ee70ec23546b76626bd8718e4b",
        "file_bytes": "3334",
        "normalized_bytes": "3638",
        "params": "10",
        "memory_bytes": "9092",
        "cost": "9102",
        "simple_cost": "3648",
        "local_points": "15.883750551423413",
        "validation_status": "265_pass_0_fail",
        "validation_pass": "6",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "9102",
        "raw_sha256": "201f50b9769bf46022aeebbb31c47469dbff8cc35b1c85b4a593daaa2c9e77b0",
        "raw_path": "experiments\\exp173_next4n_candidate_validation_audit\\task003_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 9102,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  },
  {
    "task_id": 38,
    "source_rows": 33,
    "unique_raw_candidates": 1,
    "full_ok_candidates": 1,
    "best_candidates": [
      {
        "task_id": "38",
        "filename": "task038.onnx",
        "source_label": "massimilianoghiotto_6254",
        "source_ref": "massimilianoghiotto/neurogolf2026-6254",
        "relative_path": "submission\\task038.onnx",
        "sha256": "a83d43302a9ef7e5b284cc1e38fc4ce8deedb152481de569220ea69b94855018",
        "file_bytes": "2357",
        "normalized_bytes": "2630",
        "params": "77",
        "memory_bytes": "2116",
        "cost": "2193",
        "simple_cost": "2707",
        "local_points": "17.306974251582112",
        "validation_status": "266_pass_0_fail",
        "validation_pass": "6",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "2193",
        "raw_sha256": "e17eaa6c4b6a15452543c4f573989d8aa55d3e2f0c49b95bdf2917aee713ad7c",
        "raw_path": "experiments\\exp173_next4n_candidate_validation_audit\\task038_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 2193,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  },
  {
    "task_id": 1,
    "source_rows": 33,
    "unique_raw_candidates": 2,
    "full_ok_candidates": 2,
    "best_candidates": [
      {
        "task_id": "1",
        "filename": "task001.onnx",
        "source_label": "vyanktesh_multi_source_output",
        "source_ref": "vyankteshdwivedi/neurogolf-multi-source-onnx-solver",
        "relative_path": "submission.zip::task001.onnx",
        "sha256": "80f114c73f22f7ad157c399acec303cc928f614e6e5dfb9967b3a921c1f641e2",
        "file_bytes": "1082",
        "normalized_bytes": "1319",
        "params": "24",
        "memory_bytes": "2819",
        "cost": "2843",
        "simple_cost": "1343",
        "local_points": "17.047384888349",
        "validation_status": "268_pass_0_fail",
        "validation_pass": "8",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "2843",
        "raw_sha256": "0ba1c190a016805c5f9c036c9dafac06035513bf11cea64837d053da1792a1b6",
        "raw_path": "experiments\\exp173_next4n_candidate_validation_audit\\task001_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 2843,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "1",
        "cost": "24",
        "source": "beicicc_6645",
        "exp": "exp130_public_code_6285_floor",
        "manifest": "experiments\\exp130_public_code_6285_floor\\selected_manifest.csv",
        "manifest_cost": "24",
        "raw_sha256": "1b32dc418dd9ea7c4125af3baeeeed386c4c021fa0ae77456271fb1e56b0cd19",
        "raw_path": "experiments\\exp173_next4n_candidate_validation_audit\\task001_02_exp130_public_code_6285_floor.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "validation_status": "268_pass_0_fail",
        "official_cost": 4487,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  },
  {
    "task_id": 86,
    "source_rows": 35,
    "unique_raw_candidates": 5,
    "full_ok_candidates": 4,
    "best_candidates": [
      {
        "task_id": "86",
        "route": "sparse_edit_or_object_completion",
        "template_name": "logic_Less_node16_to_input0",
        "baseline_cost": "25713",
        "candidate_cost": "25569",
        "baseline_points": "14.845248020433008",
        "candidate_points": "14.850864040809594",
        "file_bytes": "4437",
        "validation_status": "266_pass_0_fail",
        "status": "improved",
        "reason": "ok",
        "sha256": "2049bd6fbae072463ed7e40466eb3ec8fd78c544c526a1b0b0f782b0f01681f6",
        "exp": "exp034_redundant_logic_surgery_sweep",
        "manifest": "experiments\\exp034_redundant_logic_surgery_sweep\\selected_manifest.csv",
        "manifest_cost": "25569",
        "raw_sha256": "2049bd6fbae072463ed7e40466eb3ec8fd78c544c526a1b0b0f782b0f01681f6",
        "raw_path": "experiments\\exp173_next4n_candidate_validation_audit\\task086_03_exp034_redundant_logic_surgery_sweep.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 25569,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "86",
        "route": "sparse_edit_or_object_completion",
        "template_name": "redundant_and_node17_to_input0",
        "baseline_cost": "25857",
        "candidate_cost": "25713",
        "baseline_points": "14.83966336368449",
        "candidate_points": "14.845248020433008",
        "file_bytes": "4503",
        "validation_status": "266_pass_0_fail",
        "status": "improved",
        "reason": "ok",
        "sha256": "eb81a59faed292f56c40ae939a8e31c4554cb4910fe448e56899f65e9d66df2b",
        "exp": "exp033_redundant_and_surgery_sweep",
        "manifest": "experiments\\exp033_redundant_and_surgery_sweep\\selected_manifest.csv",
        "manifest_cost": "25713",
        "raw_sha256": "eb81a59faed292f56c40ae939a8e31c4554cb4910fe448e56899f65e9d66df2b",
        "raw_path": "experiments\\exp173_next4n_candidate_validation_audit\\task086_02_exp033_redundant_and_surgery_sweep.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 25713,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "86",
        "filename": "task086.onnx",
        "source_label": "massimilianoghiotto_6254",
        "source_ref": "massimilianoghiotto/neurogolf2026-6254",
        "relative_path": "submission\\task086.onnx",
        "sha256": "a786a589e429892c1114751b2e5d6b83669038bb955a0fd41d08d037fc8dc020",
        "file_bytes": "3099",
        "normalized_bytes": "4566",
        "params": "225",
        "memory_bytes": "25632",
        "cost": "25857",
        "simple_cost": "4791",
        "local_points": "14.83966336368449",
        "validation_status": "266_pass_0_fail",
        "validation_pass": "6",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "25857",
        "raw_sha256": "534022b42915e7a94b488ed3ed7c4328b1c6d323e88487a2b341cf923973ff04",
        "raw_path": "experiments\\exp173_next4n_candidate_validation_audit\\task086_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 25857,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "86",
        "cost": "25857",
        "points": "14.839663",
        "exp": "exp_b035_new_source_full_arc_blend",
        "manifest": "experiments\\exp_b035_new_source_full_arc_blend\\selected_manifest.csv",
        "manifest_cost": "25857",
        "raw_sha256": "a786a589e429892c1114751b2e5d6b83669038bb955a0fd41d08d037fc8dc020",
        "raw_path": "experiments\\exp173_next4n_candidate_validation_audit\\task086_05_exp_b035_new_source_full_arc_blend.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "validation_status": "266_pass_0_fail",
        "official_cost": 25857,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  }
]
```

## 判断

If exp171 identifies a public-zero target, use this audit to pick a distinct full_ok repair candidate before rule mining.

## リスク

medium-to-high: existing public/teacher candidates may be lookup-like.
medium-to-high: full local validation is not sufficient for public/private robustness.
