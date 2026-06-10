# exp191_risk_next4g_candidate_validation_audit

## 目的

exp190対象 `238/112/377/177` について、既存source候補のfull validation / costを事前監査する。

## 結果

- targets: `[238, 112, 377, 177]`
- full_ok_total: `6`

```json
[
  {
    "task_id": 238,
    "source_rows": 34,
    "unique_raw_candidates": 5,
    "full_ok_candidates": 2,
    "best_candidates": [
      {
        "task_id": "238",
        "filename": "task238.onnx",
        "source_label": "massimilianoghiotto_6254",
        "source_ref": "massimilianoghiotto/neurogolf2026-6254",
        "relative_path": "submission\\task238.onnx",
        "sha256": "3208dbc8cec5a6a6874fb45bdf74920c0447b6b2745454a2ca76efdd5aa83d45",
        "file_bytes": "11927",
        "normalized_bytes": "15947",
        "params": "1467",
        "memory_bytes": "62905",
        "cost": "64372",
        "simple_cost": "17414",
        "local_points": "13.927565965061614",
        "validation_status": "266_pass_0_fail",
        "validation_pass": "6",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "raw_sha256": "4a53c4b59d26583016d408d00a8065dd32544bc1b31de8279187360039ab22b9",
        "raw_path": "experiments\\exp191_risk_next4g_candidate_validation_audit\\task238_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 64372,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "238",
        "cost": "64372",
        "points": "13.927566",
        "exp": "exp_b035_new_source_full_arc_blend",
        "manifest": "experiments\\exp_b035_new_source_full_arc_blend\\selected_manifest.csv",
        "raw_sha256": "3208dbc8cec5a6a6874fb45bdf74920c0447b6b2745454a2ca76efdd5aa83d45",
        "raw_path": "experiments\\exp191_risk_next4g_candidate_validation_audit\\task238_05_exp_b035_new_source_full_arc_blend.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "validation_status": "266_pass_0_fail",
        "official_cost": 64372,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  },
  {
    "task_id": 112,
    "source_rows": 34,
    "unique_raw_candidates": 4,
    "full_ok_candidates": 1,
    "best_candidates": [
      {
        "task_id": "112",
        "filename": "task112.onnx",
        "source_label": "massimilianoghiotto_6254",
        "source_ref": "massimilianoghiotto/neurogolf2026-6254",
        "relative_path": "submission\\task112.onnx",
        "sha256": "c9d107e03bcc1559d05367b27fca4117a92b4a4e65d2103d0e57fd16eebe58ca",
        "file_bytes": "3987",
        "normalized_bytes": "5493",
        "params": "106",
        "memory_bytes": "59164",
        "cost": "59270",
        "simple_cost": "5599",
        "local_points": "14.010141445218116",
        "validation_status": "266_pass_0_fail",
        "validation_pass": "6",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "raw_sha256": "2f1ab80b5df4bad3d09c4ba3e9edd8109ec1f1e2527e8655b5c9322e99782b9d",
        "raw_path": "experiments\\exp191_risk_next4g_candidate_validation_audit\\task112_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 59270,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  },
  {
    "task_id": 377,
    "source_rows": 34,
    "unique_raw_candidates": 5,
    "full_ok_candidates": 2,
    "best_candidates": [
      {
        "task_id": "377",
        "filename": "task377.onnx",
        "source_label": "massimilianoghiotto_6254",
        "source_ref": "massimilianoghiotto/neurogolf2026-6254",
        "relative_path": "submission\\task377.onnx",
        "sha256": "9b52b78bd3e2bfa2d9837ed1edc28b7aea7bbfa859063f82428bbf9c175dfa43",
        "file_bytes": "18685",
        "normalized_bytes": "22294",
        "params": "84",
        "memory_bytes": "61840",
        "cost": "61924",
        "simple_cost": "22378",
        "local_points": "13.966336894339644",
        "validation_status": "266_pass_0_fail",
        "validation_pass": "6",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "raw_sha256": "8dab539e1f1b3c17371999dc9dc1797f79787b29423d72d417ff85b00a2a9030",
        "raw_path": "experiments\\exp191_risk_next4g_candidate_validation_audit\\task377_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 61924,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "377",
        "cost": "61924",
        "points": "13.966337",
        "exp": "exp_b035_new_source_full_arc_blend",
        "manifest": "experiments\\exp_b035_new_source_full_arc_blend\\selected_manifest.csv",
        "raw_sha256": "9b52b78bd3e2bfa2d9837ed1edc28b7aea7bbfa859063f82428bbf9c175dfa43",
        "raw_path": "experiments\\exp191_risk_next4g_candidate_validation_audit\\task377_05_exp_b035_new_source_full_arc_blend.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "validation_status": "266_pass_0_fail",
        "official_cost": 61924,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  },
  {
    "task_id": 177,
    "source_rows": 34,
    "unique_raw_candidates": 4,
    "full_ok_candidates": 1,
    "best_candidates": [
      {
        "task_id": "177",
        "filename": "task177.onnx",
        "source_label": "massimilianoghiotto_6254",
        "source_ref": "massimilianoghiotto/neurogolf2026-6254",
        "relative_path": "submission\\task177.onnx",
        "sha256": "cf49f3845c907c5c58bfc44858a09562b29c11f7bc05ec633f96e70552287949",
        "file_bytes": "3189",
        "normalized_bytes": "3685",
        "params": "102",
        "memory_bytes": "61092",
        "cost": "61194",
        "simple_cost": "3787",
        "local_points": "13.978195575521427",
        "validation_status": "265_pass_0_fail",
        "validation_pass": "6",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "raw_sha256": "50dc5459fed670dc33d25ef8c29dd8edb245edf554ee923c46c750b8fe65edbf",
        "raw_path": "experiments\\exp191_risk_next4g_candidate_validation_audit\\task177_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 61194,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  }
]
```

## 判断

Use this audit if exp190 identifies a public-zero target among task238/112/377/177.

## リスク

medium-to-high: existing public/teacher candidates may be lookup-like.
medium-to-high: full local validation is not sufficient for public/private robustness.
