# exp010_cuda_gpu_setup notes

## 目的

GPU学習をtemplate routingと候補rankingに使うため、既存 `.venv` にCUDA版 `torch + torchvision` を導入する。

## 結果

- status: `cuda_ready`
- `torch`: `2.12.0+cu126`
- `torchvision`: `0.27.0+cu126`
- GPU: `NVIDIA GeForce RTX 2080 SUPER`
- CUDA available: `true`
- small matmul: pass
- small CNN forward/backward: pass
- smoke test最大割当: `29.85 MB`

## 判断

CUDA導入は成功。GPUは最終提出ONNXには直接使わず、route分類器とtemplate候補rankingの補助に使う。

## Risk

- leakage risk: 中。route classifierはteacher labelをrule-based taxonomyから作るため、採用判定には使わず候補生成順に限定する。
- overfitting risk: 中。task数が400と少ないため、task_id group splitで監視する。
