# exp129 compiler farm core

This experiment adds the reusable core for a NeuroGolf ONNX optimization farm.

- `experiments/neurogolf_farm/ir.py`: small NeuroGolf-specific IR before ONNX emission.
- `experiments/neurogolf_farm/cost_extractor.py`: pre-emission guardrails for high-cost ONNX patterns.
- `experiments/neurogolf_farm/public_code.py`: public CODE registry and 6285 floor adoption gates.
- `experiments/neurogolf_farm/bundle_manager.py`: accepted-candidate ledger and bundle adoption gates.
- `experiments/neurogolf_farm/farm_runner.py`: smoke runner tying guardrails and public-code intake together.

The key policy is conservative: public CODE can raise the submission floor only after full-arc validation and Kaggle LB evidence. Until then it is teacher/intelligence for rule mining, lowering design, and candidate prioritization.
