# exp010_cuda_gpu_setup

## 目的

CUDA版 PyTorch を既存 `.venv` に導入し、GPU学習をtemplate routingと候補rankingに使える状態にする。

## 結果

- status: `cuda_ready`
- Python: `3.12.10`
- torch: `2.12.0+cu126`
- torchvision: `0.27.0+cu126`
- GPU: `NVIDIA GeForce RTX 2080 SUPER`
- CUDA available: `true`
- small matmul: pass
- small CNN forward/backward: pass

## 判断

GPUは使用可能。提出ONNXそのものには使わず、route分類・特徴抽出・候補rankingの補助に使う。

## Risk

- CUDA wheelは公式PyTorch index依存なので、`requirements.txt` に再現用indexを残す。
- `model.pt` や生成zipはGit管理外にする。
