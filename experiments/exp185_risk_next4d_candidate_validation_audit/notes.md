# exp185_risk_next4d_candidate_validation_audit

## 目的

exp183対象 `187/204/198/364` について、既存source候補のfull validation / costを監査する。特にtask187のpublic-zero repair候補を探す。

## 結果

- targets: `[187, 204, 198, 364]`
- full_ok_total: `4`

```json
[
  {
    "task_id": 187,
    "source_rows": 37,
    "unique_raw_candidates": 6,
    "full_ok_candidates": 1,
    "best_candidates": [
      {
        "task_id": "187",
        "filename": "task187.onnx",
        "source_label": "massimilianoghiotto_6254",
        "source_ref": "massimilianoghiotto/neurogolf2026-6254",
        "relative_path": "submission\\task187.onnx",
        "sha256": "a12ff26344d3636a415f318272f96b57a7e6c98ebd9e5bd92bc9465f8ff342d7",
        "file_bytes": "5758",
        "normalized_bytes": "7916",
        "params": "913",
        "memory_bytes": "104400",
        "cost": "105313",
        "simple_cost": "8829",
        "local_points": "13.435307852707972",
        "validation_status": "266_pass_0_fail",
        "validation_pass": "6",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "105313",
        "raw_sha256": "2ebbc64c3257fa66b2030c42845bb1b9da22d09c9369fe61e530a138b97cb46a",
        "raw_path": "experiments\\exp185_risk_next4d_candidate_validation_audit\\task187_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 105313,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  },
  {
    "task_id": 204,
    "source_rows": 34,
    "unique_raw_candidates": 5,
    "full_ok_candidates": 1,
    "best_candidates": [
      {
        "task_id": "204",
        "filename": "task204.onnx",
        "source_label": "kojimar_5800_minimal_blend",
        "source_ref": "kojimar/neurogolf-5800-55-minimal-onnx-blend-assets",
        "relative_path": "base_submission\\task204.onnx",
        "sha256": "6c21636d516c553f675736fe99ab464e08d17040686f3354615b8e54bb26a472",
        "file_bytes": "4457",
        "normalized_bytes": "5559",
        "params": "1044",
        "memory_bytes": "103500",
        "cost": "104544",
        "simple_cost": "6603",
        "local_points": "13.442636685599203",
        "validation_status": "268_pass_0_fail",
        "validation_pass": "8",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "104544",
        "raw_sha256": "21b9b3f65c96474ed96fb4096e2cbd4ada67cb1d713bcd4af009dda771a797ea",
        "raw_path": "experiments\\exp185_risk_next4d_candidate_validation_audit\\task204_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 104544,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  },
  {
    "task_id": 198,
    "source_rows": 33,
    "unique_raw_candidates": 2,
    "full_ok_candidates": 1,
    "best_candidates": [
      {
        "task_id": "198",
        "filename": "task198.onnx",
        "source_label": "massimilianoghiotto_6254",
        "source_ref": "massimilianoghiotto/neurogolf2026-6254",
        "relative_path": "submission\\task198.onnx",
        "sha256": "a1cd099e3d965fb4506eded8d4bde77ea202357df70e9212b36911ca002c8cb1",
        "file_bytes": "6979",
        "normalized_bytes": "8402",
        "params": "913",
        "memory_bytes": "102908",
        "cost": "103821",
        "simple_cost": "9315",
        "local_points": "13.449576458609688",
        "validation_status": "266_pass_0_fail",
        "validation_pass": "6",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "103821",
        "raw_sha256": "1466ce35887bc1eb4d290c46e10e5dfd67f58aeb1bf7ccdf113d625ccd8ff679",
        "raw_path": "experiments\\exp185_risk_next4d_candidate_validation_audit\\task198_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 103821,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  },
  {
    "task_id": 364,
    "source_rows": 34,
    "unique_raw_candidates": 4,
    "full_ok_candidates": 1,
    "best_candidates": [
      {
        "task_id": "364",
        "filename": "task364.onnx",
        "source_label": "massimilianoghiotto_6254",
        "source_ref": "massimilianoghiotto/neurogolf2026-6254",
        "relative_path": "submission\\task364.onnx",
        "sha256": "9f5bdb27f1d3209b0218aec767937d478bbe7db43fdf2be16278461b10f9c75f",
        "file_bytes": "8985",
        "normalized_bytes": "10192",
        "params": "963",
        "memory_bytes": "76896",
        "cost": "77859",
        "simple_cost": "11155",
        "local_points": "13.737345222483407",
        "validation_status": "266_pass_0_fail",
        "validation_pass": "6",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "77859",
        "raw_sha256": "a150d169b9c93d74c89bc493a9d92d7b587d4a317114f5f2124ccb6fb79e222d",
        "raw_path": "experiments\\exp185_risk_next4d_candidate_validation_audit\\task364_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 77859,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  }
]
```

## 判断

Use this audit to pick a distinct full_ok repair candidate for task187 if exp183 identifies task187 as public-zero.

## リスク

medium-to-high: existing public/teacher candidates may be lookup-like.
medium-to-high: full local validation is not sufficient for public/private robustness.
