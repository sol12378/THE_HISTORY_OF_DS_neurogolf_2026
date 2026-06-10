# exp180_risk_next4_candidate_validation_audit

## 目的

exp179対象 `251/109/239/358` について、既存source候補のfull validation / costを事前監査する。

## 結果

- targets: `[251, 109, 239, 358]`
- full_ok_total: `8`

```json
[
  {
    "task_id": 251,
    "source_rows": 34,
    "unique_raw_candidates": 6,
    "full_ok_candidates": 2,
    "best_candidates": [
      {
        "task_id": "251",
        "filename": "task251.onnx",
        "source_label": "massimilianoghiotto_6254",
        "source_ref": "massimilianoghiotto/neurogolf2026-6254",
        "relative_path": "submission\\task251.onnx",
        "sha256": "82d350374d90693c70187cdf979ebb574c69cda11c9c56813abeb372eed8e78f",
        "file_bytes": "5106",
        "normalized_bytes": "6781",
        "params": "20",
        "memory_bytes": "100560",
        "cost": "100580",
        "simple_cost": "6801",
        "local_points": "13.481291290274044",
        "validation_status": "266_pass_0_fail",
        "validation_pass": "6",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "100580",
        "raw_sha256": "474eb593e33a540db6438a2041d5c2747c44f4f5f49a9d883147e2716d5e8a66",
        "raw_path": "experiments\\exp180_risk_next4_candidate_validation_audit\\task251_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 100580,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "251",
        "cost": "100580",
        "points": "13.481291",
        "exp": "exp_b035_new_source_full_arc_blend",
        "manifest": "experiments\\exp_b035_new_source_full_arc_blend\\selected_manifest.csv",
        "manifest_cost": "100580",
        "raw_sha256": "82d350374d90693c70187cdf979ebb574c69cda11c9c56813abeb372eed8e78f",
        "raw_path": "experiments\\exp180_risk_next4_candidate_validation_audit\\task251_06_exp_b035_new_source_full_arc_blend.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "validation_status": "266_pass_0_fail",
        "official_cost": 100580,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  },
  {
    "task_id": 109,
    "source_rows": 34,
    "unique_raw_candidates": 6,
    "full_ok_candidates": 2,
    "best_candidates": [
      {
        "task_id": "109",
        "filename": "task109.onnx",
        "source_label": "massimilianoghiotto_6254",
        "source_ref": "massimilianoghiotto/neurogolf2026-6254",
        "relative_path": "submission\\task109.onnx",
        "sha256": "d2ade70f5e497c980bfbb088e8cc977d66bc76830c18c7e11004e7eddfa1f726",
        "file_bytes": "5257",
        "normalized_bytes": "5912",
        "params": "954",
        "memory_bytes": "93924",
        "cost": "94878",
        "simple_cost": "6866",
        "local_points": "13.539652865248629",
        "validation_status": "266_pass_0_fail",
        "validation_pass": "6",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "94878",
        "raw_sha256": "77772c80a0102b55408b00b92b38215ee4075c4753854a4eb9c779ec362b9487",
        "raw_path": "experiments\\exp180_risk_next4_candidate_validation_audit\\task109_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 94878,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "109",
        "cost": "184622",
        "points": "12.873934",
        "exp": "exp_b035_new_source_full_arc_blend",
        "manifest": "experiments\\exp_b035_new_source_full_arc_blend\\selected_manifest.csv",
        "manifest_cost": "184622",
        "raw_sha256": "30bb537b9f58dd62c94bccda3cb764766feb3a061e5b5596e2ef7cb7af6f12f9",
        "raw_path": "experiments\\exp180_risk_next4_candidate_validation_audit\\task109_06_exp_b035_new_source_full_arc_blend.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "validation_status": "266_pass_0_fail",
        "official_cost": 184622,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  },
  {
    "task_id": 239,
    "source_rows": 34,
    "unique_raw_candidates": 5,
    "full_ok_candidates": 1,
    "best_candidates": [
      {
        "task_id": "239",
        "filename": "task239.onnx",
        "source_label": "afr1ste_5689_artifact",
        "source_ref": "afr1ste/neurogolf-5689-51-current-rules-open-artifact",
        "relative_path": "submission\\task239.onnx",
        "sha256": "1bb32963e7cb07aed708c1c9c2e30991506b48570898ddade7fec2515559fc95",
        "file_bytes": "3146",
        "normalized_bytes": "3677",
        "params": "386",
        "memory_bytes": "94091",
        "cost": "94477",
        "simple_cost": "4063",
        "local_points": "13.54388830238479",
        "validation_status": "267_pass_0_fail",
        "validation_pass": "7",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "94477",
        "raw_sha256": "000e1472771006e984fdc1ef874767713c8679fd86abbb5839ad51ebe693b9e2",
        "raw_path": "experiments\\exp180_risk_next4_candidate_validation_audit\\task239_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 94477,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  },
  {
    "task_id": 358,
    "source_rows": 34,
    "unique_raw_candidates": 7,
    "full_ok_candidates": 3,
    "best_candidates": [
      {
        "task_id": "358",
        "source": "exp068_seddik_style_strict_scalarization_seddik_style_scalarize_dedup_prune",
        "template_name": "seddik_style_scalarize_dedup_prune",
        "route": "sparse_edit_or_object_completion",
        "cost": "113921",
        "local_points": "13.356739495302836",
        "file_bytes": "11508",
        "status": "improved",
        "reason": "ok",
        "sha256": "9fb2a68b39498bad077ae912a10f201291580921463e675cef65706d1f0ee0b0",
        "exp": "exp068_seddik_style_strict_scalarization",
        "manifest": "experiments\\exp068_seddik_style_strict_scalarization\\selected_manifest.csv",
        "manifest_cost": "113921",
        "raw_sha256": "9fb2a68b39498bad077ae912a10f201291580921463e675cef65706d1f0ee0b0",
        "raw_path": "experiments\\exp180_risk_next4_candidate_validation_audit\\task358_05_exp068_seddik_style_strict_scalarization.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "validation_status": "265_pass_0_fail",
        "official_cost": 113921,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "358",
        "cost": "113943",
        "points": "13.356546",
        "exp": "exp_b035_new_source_full_arc_blend",
        "manifest": "experiments\\exp_b035_new_source_full_arc_blend\\selected_manifest.csv",
        "manifest_cost": "113943",
        "raw_sha256": "0d7fb16add136afdbf237c35e2ced3c28e9f36f9e75bc94eb7fd6f0703067cda",
        "raw_path": "experiments\\exp180_risk_next4_candidate_validation_audit\\task358_07_exp_b035_new_source_full_arc_blend.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "validation_status": "265_pass_0_fail",
        "official_cost": 113943,
        "score_reason": "ok",
        "submit_candidate": true
      },
      {
        "task_id": "358",
        "filename": "task358.onnx",
        "source_label": "afr1ste_5689_artifact",
        "source_ref": "afr1ste/neurogolf-5689-51-current-rules-open-artifact",
        "relative_path": "submission\\task358.onnx",
        "sha256": "cd5df694270b443fd67739a2c4861762e269841a08d214644221206222ef004b",
        "file_bytes": "6755",
        "normalized_bytes": "11696",
        "params": "415",
        "memory_bytes": "113584",
        "cost": "113999",
        "simple_cost": "12111",
        "local_points": "13.356055044591665",
        "validation_status": "265_pass_0_fail",
        "validation_pass": "5",
        "validation_fail": "0",
        "exp": "exp002_public_blend_6500_fast",
        "manifest": "experiments\\exp002_public_blend_6500_fast\\selected_manifest.csv",
        "manifest_cost": "113999",
        "raw_sha256": "54402e388ac2926df1ac66113424b75a4f8b0455364747db30fbd17754b55843",
        "raw_path": "experiments\\exp180_risk_next4_candidate_validation_audit\\task358_01_exp002_public_blend_6500_fast.onnx",
        "audit_status": "full_ok",
        "audit_reason": "ok",
        "official_cost": 113999,
        "score_reason": "ok",
        "submit_candidate": true
      }
    ]
  }
]
```

## 判断

If exp179 identifies a public-zero target, use this audit to pick a distinct full_ok repair candidate before rule mining.

## リスク

medium-to-high: existing public/teacher candidates may be lookup-like.
medium-to-high: full local validation is not sufficient for public/private robustness.
